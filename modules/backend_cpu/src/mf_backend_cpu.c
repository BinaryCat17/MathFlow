#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_state.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_thread_pool.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_platform.h>
#include <stdlib.h>
#include <string.h>

// --- Constants ---

#define MF_CPU_JOB_SIZE         4096         // Elements per job (Linear)
#define MF_CPU_INLINE_THRESHOLD 1024         // If total elements < this, run inline
#define MF_CPU_WORKER_HEAP_SZ   (16*1024*1024) // 16MB per worker
#define MF_CPU_REG_ARENA_SZ     8192         // 8KB for registers metadata
#define MF_MAX_REGISTERS        512          // Max tensors per program

// --- Internal Structures ---

typedef enum {
    MF_TENSOR_ROLE_UNIFORM,
    MF_TENSOR_ROLE_SPATIAL,
} mf_tensor_role;

typedef struct {
    mf_thread_pool* pool;
    mf_op_func op_table[MF_OP_LIMIT];
} mf_backend_cpu_state;

typedef struct {
    mf_exec_ctx ctx;
    mf_arena temp_arena; 
    void* heap_mem;
    size_t heap_size;
    mf_arena reg_arena;
    u8 reg_arena_mem[MF_CPU_REG_ARENA_SZ]; 
} mf_backend_cpu_worker_state;

typedef struct {
    const mf_program* program;
    mf_state* main_state;
    mf_op_func* op_table;
    
    size_t total_elements;
    u8 ndim;
    u32 domain_shape[MF_MAX_DIMS];
    mf_tensor_role roles[MF_MAX_REGISTERS];
    int channels[MF_MAX_REGISTERS];
} mf_cpu_parallel_batch;

// --- Worker Lifecycle ---

static void* worker_init(int thread_idx, void* user_data) {
    (void)thread_idx; (void)user_data;
    mf_backend_cpu_worker_state* state = malloc(sizeof(mf_backend_cpu_worker_state));
    state->heap_mem = malloc(MF_CPU_WORKER_HEAP_SZ);
    state->heap_size = MF_CPU_WORKER_HEAP_SZ;
    mf_arena_init(&state->temp_arena, state->heap_mem, state->heap_size);
    mf_arena_init(&state->reg_arena, state->reg_arena_mem, sizeof(state->reg_arena_mem));
    return state;
}

static void worker_cleanup(void* thread_local_data, void* user_data) {
    (void)user_data;
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    if (!state) return;
    free(state->heap_mem);
    free(state);
}

// --- Execution Logic ---

static inline void mf_cpu_exec(mf_exec_ctx* ctx, const mf_program* program, mf_op_func* op_table) {
    const size_t code_count = program->meta.instruction_count;
    const mf_instruction* code = program->code;
    
    for (size_t i = 0; i < code_count; ++i) {
        if (ctx->error != MF_ERROR_NONE) break;
        const mf_instruction* inst = &code[i];
        mf_op_func op = op_table[inst->opcode];
        if (op) op(ctx, inst);
    }
}

static void prepare_registers(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch, size_t start_idx, size_t count) {
    mf_exec_ctx* worker_ctx = &state->ctx;
    mf_state* main_state = batch->main_state;
    
    for (size_t i = 0; i < worker_ctx->register_count; ++i) {
        mf_tensor* main_t = &main_state->registers[i];
        mf_tensor* worker_t = &worker_ctx->registers[i];

        // Start with a full copy
        *worker_t = *main_t;

        if (batch->roles[i] == MF_TENSOR_ROLE_SPATIAL && mf_tensor_is_valid(main_t)) {
            size_t dtype_sz = mf_dtype_size(main_t->info.dtype);
            worker_t->byte_offset = main_t->byte_offset + (start_idx * dtype_sz * (size_t)batch->channels[i]);
            
            // Adjust metadata for the flat window view
            worker_t->info.ndim = (batch->channels[i] > 1) ? 2 : 1;
            worker_t->info.shape[0] = (int32_t)count;
            worker_t->info.strides[0] = batch->channels[i];
            if (batch->channels[i] > 1) {
                worker_t->info.shape[1] = batch->channels[i];
                worker_t->info.strides[1] = 1;
            }
        }
    }
}

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    mf_cpu_parallel_batch* batch = (mf_cpu_parallel_batch*)user_data;

    size_t start_idx = (size_t)job_idx * MF_CPU_JOB_SIZE;
    size_t count = MF_CPU_JOB_SIZE;
    if (start_idx + count > batch->total_elements) {
        count = batch->total_elements - start_idx;
    }

    if (count == 0) return;

    mf_arena_reset(&state->reg_arena);
    mf_arena_reset(&state->temp_arena);
    
    u32 reg_count = batch->program->meta.tensor_count;
    mf_tensor* local_regs = MF_ARENA_PUSH(&state->reg_arena, mf_tensor, reg_count);
    mf_exec_ctx_init(&state->ctx, local_regs, reg_count, (mf_allocator*)&state->temp_arena);
    
    state->ctx.batch_size = (u32)count;
    state->ctx.ndim = batch->ndim; 
    state->ctx.tile_offset[0] = (u32)start_idx; 

    // Unflatten start index for N-dimensional operations (e.g. op_index)
    size_t temp_idx = start_idx;
    for (int i = batch->ndim - 1; i >= 0; --i) {
        state->ctx.tile_offset[i] = (u32)(temp_idx % batch->domain_shape[i]);
        temp_idx /= batch->domain_shape[i];
    }
    // Restore linear offset to tile_offset[0] as expected by existing op_index implementation
    state->ctx.tile_offset[0] = (u32)start_idx;

    for(int d=0; d<batch->ndim; ++d) state->ctx.domain_shape[d] = batch->domain_shape[d];

    prepare_registers(state, batch, start_idx, count);

    mf_cpu_exec(&state->ctx, batch->program, batch->op_table);
    
    if (state->ctx.error != MF_ERROR_NONE && batch->main_state) {
        mf_atomic_store((mf_atomic_i32*)&batch->main_state->error_code, (int32_t)state->ctx.error);
    }
}

// --- Dispatch ---

static void mf_backend_cpu_dispatch(
    void* backend_state,
    const struct mf_program* program,
    struct mf_state* main_state,
    const mf_tensor* domain
) {
    mf_backend_cpu_state* state = (mf_backend_cpu_state*)backend_state;
    if (!domain) return;

    size_t total_elements = mf_tensor_count(domain);
    if (total_elements == 0) return;

    mf_cpu_parallel_batch batch = {
        .program = program,
        .main_state = main_state,
        .op_table = state->op_table,
        .total_elements = total_elements,
        .ndim = domain->info.ndim,
    };
    memcpy(batch.domain_shape, domain->info.shape, sizeof(u32) * MF_MAX_DIMS);

    // Pre-calculate tensor roles to avoid expensive checks in workers
    for (u32 i = 0; i < program->meta.tensor_count; ++i) {
        mf_tensor* main_t = &main_state->registers[i];
        batch.roles[i] = MF_TENSOR_ROLE_UNIFORM;
        batch.channels[i] = 1;

        if (!mf_tensor_is_valid(main_t)) continue;

        size_t t_count = mf_tensor_count(main_t);
        if (t_count == total_elements) {
            batch.roles[i] = MF_TENSOR_ROLE_SPATIAL;
            batch.channels[i] = 1;
        } else if (total_elements > 0 && t_count > total_elements && t_count % total_elements == 0) {
            batch.roles[i] = MF_TENSOR_ROLE_SPATIAL;
            batch.channels[i] = (int)(t_count / total_elements);
        }
    }

    u32 total_jobs = (u32)((total_elements + MF_CPU_JOB_SIZE - 1) / MF_CPU_JOB_SIZE);

    if (total_elements <= MF_CPU_INLINE_THRESHOLD || total_jobs == 1) {
        mf_backend_cpu_worker_state local_worker;
        u8 local_heap[MF_KB(64)]; 
        local_worker.heap_mem = local_heap;
        local_worker.heap_size = sizeof(local_heap);
        mf_arena_init(&local_worker.temp_arena, local_worker.heap_mem, local_worker.heap_size);
        mf_arena_init(&local_worker.reg_arena, local_worker.reg_arena_mem, sizeof(local_worker.reg_arena_mem));
        
        cpu_worker_job(0, &local_worker, &batch);
        return;
    }

    if (state->pool) {
        mf_thread_pool_run(state->pool, total_jobs, cpu_worker_job, &batch);
    } else {
        // Fallback for systems without thread pool support
        void* persistent_worker = worker_init(0, NULL);
        for (u32 i = 0; i < total_jobs; ++i) {
            cpu_worker_job(i, persistent_worker, &batch);
        }
        worker_cleanup(persistent_worker, NULL);
    }
}

static void mf_backend_cpu_shutdown(void* backend_state) {
    mf_backend_cpu_state* state = (mf_backend_cpu_state*)backend_state;
    if (!state) return;
    if (state->pool) mf_thread_pool_destroy(state->pool);
    free(state);
}

void mf_backend_cpu_init(mf_backend* backend, int num_threads) {
    memset(backend, 0, sizeof(mf_backend));
    mf_backend_cpu_state* state = calloc(1, sizeof(mf_backend_cpu_state));
    mf_thread_pool_desc pool_desc = {
        .num_threads = num_threads,
        .init_fn = worker_init,
        .cleanup_fn = worker_cleanup
    };
    state->pool = mf_thread_pool_create(&pool_desc);
    mf_ops_fill_table(state->op_table);
    backend->state = state;
    backend->shutdown = mf_backend_cpu_shutdown;
    backend->dispatch = mf_backend_cpu_dispatch;
}