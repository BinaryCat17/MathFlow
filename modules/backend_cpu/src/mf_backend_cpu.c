#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_state.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_thread_pool.h>
#include <mathflow/base/mf_log.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- Constants ---

#define MF_CPU_JOB_SIZE 4096         // Elements per job (Linear)
#define MF_CPU_INLINE_THRESHOLD 1024 // If total elements < this, run inline

// --- Internal Structures ---

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
    u8 reg_arena_mem[8192]; 
} mf_backend_cpu_worker_state;

typedef struct {
    const mf_program* program;
    mf_state* main_state;
    mf_op_func* op_table;
    
    size_t total_elements;
    u8 ndim;
    u32 domain_shape[MF_MAX_DIMS];
} mf_cpu_parallel_batch;

// --- Worker Lifecycle ---

static void* worker_init(int thread_idx, void* user_data) {
    (void)thread_idx; (void)user_data;
    mf_backend_cpu_worker_state* state = malloc(sizeof(mf_backend_cpu_worker_state));
    size_t heap_size = 16 * 1024 * 1024;
    state->heap_mem = malloc(heap_size);
    state->heap_size = heap_size;
    mf_arena_init(&state->temp_arena, state->heap_mem, heap_size);
    mf_arena_init(&state->reg_arena, state->reg_arena_mem, sizeof(state->reg_arena_mem));
    return state;
}

static void worker_cleanup(void* thread_local_data, void* user_data) {
    (void)user_data;
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    free(state->heap_mem);
    free(state);
}

// --- Execution Logic ---

static void mf_cpu_exec(mf_exec_ctx* ctx, const mf_program* program, mf_op_func* op_table) {
    size_t code_count = program->meta.instruction_count;
    mf_instruction* code = program->code;
    for (size_t i = 0; i < code_count; ++i) {
        if (ctx->error != MF_ERROR_NONE) break;
        mf_instruction* inst = &code[i];
        if (op_table[inst->opcode]) {
            op_table[inst->opcode](ctx, inst);
        }
    }
}

static void prepare_registers(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch, size_t start_idx, size_t count) {
    mf_exec_ctx* worker_ctx = &state->ctx;
    mf_state* main_state = batch->main_state;
    
    for (size_t i = 0; i < worker_ctx->register_count; ++i) {
        mf_tensor* main_t = &main_state->registers[i];
        mf_tensor* worker_t = &worker_ctx->registers[i];
        
        // Default: Initialize from program descriptor (shape/type)
        *worker_t = batch->program->tensors[i];

        bool is_spatial = false;
        int channels = 1;

        // 1. Strict Match ([Domain])
        if (main_t->info.ndim == batch->ndim) {
            bool match = true;
            for(int d=0; d<batch->ndim; ++d) {
                if ((u32)main_t->info.shape[d] != batch->domain_shape[d]) { match = false; break; }
            }
            if (match) is_spatial = true;
        }
        // 2. Vector Match ([Domain, C])
        else if (main_t->info.ndim == batch->ndim + 1) {
            bool match = true;
            for(int d=0; d<batch->ndim; ++d) {
                if ((u32)main_t->info.shape[d] != batch->domain_shape[d]) { match = false; break; }
            }
            if (match) {
                is_spatial = true;
                channels = main_t->info.shape[batch->ndim];
            }
        }

        if (is_spatial && mf_tensor_is_valid(main_t)) {
            // ZERO-COPY WINDOW: Point worker tensor directly to a subset of the main buffer
            size_t dtype_sz = mf_dtype_size(main_t->info.dtype);
            
            worker_t->buffer = main_t->buffer;
            worker_t->byte_offset = main_t->byte_offset + (start_idx * dtype_sz);
            
            // Spatial tensors in worker context are ALWAYS flattened to [batch_size]
            // regardless of their original channel count, because the domain already 
            // accounts for channels.
            worker_t->info.ndim = 1;
            worker_t->info.shape[0] = (int32_t)count;
            worker_t->info.strides[0] = 1;
        }
        else {
            // 2. Uniform Propagation (Scalars or small constants)
            void* main_data = mf_tensor_data(main_t);
            if (main_data) {
                size_t bytes = mf_tensor_size_bytes(main_t);
                void* local_mem = mf_arena_alloc((mf_allocator*)&state->temp_arena, bytes);
                if (local_mem) {
                    memcpy(local_mem, main_data, bytes);
                    
                    mf_buffer* buf = mf_arena_alloc((mf_allocator*)&state->temp_arena, sizeof(mf_buffer));
                    mf_buffer_init_view(buf, local_mem, bytes);
                    worker_t->buffer = buf;
                    worker_t->byte_offset = 0;
                    worker_t->info = main_t->info;
                }
            }
        }
    }
}

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    mf_cpu_parallel_batch* batch = (mf_cpu_parallel_batch*)user_data;

    // 1. Calculate Linear Range
    size_t start_idx = (size_t)job_idx * MF_CPU_JOB_SIZE;
    size_t count = MF_CPU_JOB_SIZE;
    if (start_idx + count > batch->total_elements) {
        count = batch->total_elements - start_idx;
    }

    if (count == 0) return;

    // 2. Reset Context
    mf_arena_reset(&state->reg_arena);
    mf_arena_reset(&state->temp_arena);
    
    // 3. Setup Local Registers
    u32 reg_count = batch->program->meta.tensor_count;
    mf_tensor* local_regs = MF_ARENA_PUSH(&state->reg_arena, mf_tensor, reg_count);
    mf_exec_ctx_init(&state->ctx, local_regs, reg_count, (mf_allocator*)&state->temp_arena);
    
    // 4. Fill Context & Bind Windows
    state->ctx.batch_size = (u32)count;
    state->ctx.ndim = batch->ndim; // Logical Dimensions
    state->ctx.tile_offset[0] = (u32)start_idx; // Store Linear Offset in [0]
    // Copy logical domain shape
    for(int d=0; d<batch->ndim; ++d) state->ctx.domain_shape[d] = batch->domain_shape[d];

    prepare_registers(state, batch, start_idx, count);

    // 5. Execute
    mf_cpu_exec(&state->ctx, batch->program, batch->op_table);
    
    if (state->ctx.error != MF_ERROR_NONE && batch->main_state) {
        batch->main_state->error_code = (int32_t)state->ctx.error;
    }
    // NOTE: commit_outputs is GONE. Data is already in main buffers.
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

    MF_LOG_TRACE("CPU Dispatch: elements=%zu, ndim=%u, shape=[%u, %u, %u]", 
        total_elements, batch.ndim, batch.domain_shape[0], batch.domain_shape[1], batch.domain_shape[2]);

    u32 total_jobs = (u32)((total_elements + MF_CPU_JOB_SIZE - 1) / MF_CPU_JOB_SIZE);

    if (total_elements <= MF_CPU_INLINE_THRESHOLD || total_jobs == 1) {
        mf_backend_cpu_worker_state* temp_worker = worker_init(0, NULL);
        cpu_worker_job(0, temp_worker, &batch);
        worker_cleanup(temp_worker, NULL);
        return;
    }

    if (state->pool) {
        mf_thread_pool_run(state->pool, total_jobs, cpu_worker_job, &batch);
    } else {
        mf_backend_cpu_worker_state* temp_worker = worker_init(0, NULL);
        for (u32 i = 0; i < total_jobs; ++i) {
            cpu_worker_job(i, temp_worker, &batch);
        }
        worker_cleanup(temp_worker, NULL);
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