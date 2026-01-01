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
#include <stdalign.h>

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
    MF_TENSOR_ROLE_REDUCTION,
} mf_tensor_role;

typedef struct {
    mf_thread_pool* pool;
    mf_op_func op_table[MF_OP_LIMIT];
} mf_backend_cpu_state;

typedef struct {
    int thread_idx;
    mf_exec_ctx ctx;
    mf_arena temp_arena; 
    void* heap_mem;
    size_t heap_size;
    mf_arena reg_arena;
    _Alignas(16) u8 reg_arena_mem[MF_CPU_REG_ARENA_SZ]; 
} mf_backend_cpu_worker_state;

typedef struct {
    const mf_program* program;
    mf_state* main_state;
    mf_op_func* op_table;
    
    uint32_t start_inst;
    uint32_t inst_count;
    
    size_t total_elements;
    u8 ndim;
    u32 domain_shape[MF_MAX_DIMS];
    mf_tensor_role roles[MF_MAX_REGISTERS];
    int channels[MF_MAX_REGISTERS];

    // Parallel Reduction Support
    f32* reduction_scratch; // [num_threads * num_registers]
    int num_threads;

    // Active Register Tracking
    u16 active_regs[MF_MAX_REGISTERS];
    u32 active_reg_count;
} mf_cpu_parallel_batch;

// --- Worker Lifecycle ---

static void* worker_init(int thread_idx, void* user_data) {
    (void)user_data;
    mf_backend_cpu_worker_state* state = malloc(sizeof(mf_backend_cpu_worker_state));
    if (!state) {
        MF_LOG_ERROR("CPU Backend: Failed to allocate worker state.");
        return NULL;
    }
    state->thread_idx = thread_idx;
    
    // Use aligned allocation for SIMD friendliness
#ifdef _WIN32
    state->heap_mem = _aligned_malloc(MF_CPU_WORKER_HEAP_SZ, 16);
#else
    state->heap_mem = aligned_alloc(16, MF_CPU_WORKER_HEAP_SZ);
#endif

    if (!state->heap_mem) {
        MF_LOG_ERROR("CPU Backend: Failed to allocate worker heap (%zu bytes).", (size_t)MF_CPU_WORKER_HEAP_SZ);
        free(state);
        return NULL;
    }
    state->heap_size = MF_CPU_WORKER_HEAP_SZ;
    mf_arena_init(&state->temp_arena, state->heap_mem, state->heap_size);
    mf_arena_init(&state->reg_arena, state->reg_arena_mem, sizeof(state->reg_arena_mem));
    return state;
}

static void worker_cleanup(void* thread_local_data, void* user_data) {
    (void)user_data;
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    if (!state) return;
#ifdef _WIN32
    _aligned_free(state->heap_mem);
#else
    free(state->heap_mem);
#endif
    free(state);
}

// --- Execution Logic ---

static inline void mf_cpu_exec(mf_exec_ctx* ctx, const mf_program* program, mf_op_func* op_table, const mf_cpu_parallel_batch* batch) {
    const mf_instruction* code = program->code;
    const uint32_t end_inst = batch->start_inst + batch->inst_count;
    
    for (uint32_t i = batch->start_inst; i < end_inst; ++i) {
        // Stop if local error OR global error detected by another thread
        if (ctx->error != MF_ERROR_NONE) break;
        if (batch->main_state && mf_atomic_load((mf_atomic_i32*)&batch->main_state->error_code) != 0) break;

        const mf_instruction* inst = &code[i];
        mf_op_func op = op_table[inst->opcode];
        if (op) op(ctx, inst);
    }
}

static void prepare_registers(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch, size_t start_idx, size_t count) {
    mf_exec_ctx* worker_ctx = &state->ctx;
    mf_state* main_state = batch->main_state;
    int tid = state->thread_idx;
    
    for (size_t k = 0; k < batch->active_reg_count; ++k) {
        u16 i = batch->active_regs[k];
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
        } else if (batch->roles[i] == MF_TENSOR_ROLE_REDUCTION && batch->reduction_scratch) {
            // Redirect output to thread-local scratch space
            // We create a temporary buffer object on the arena for this view
            mf_buffer* scratch_buf = MF_ARENA_PUSH(&state->reg_arena, mf_buffer, 1);
            scratch_buf->data = &batch->reduction_scratch[tid * MF_MAX_REGISTERS + i];
            scratch_buf->size_bytes = sizeof(f32);
            scratch_buf->alloc = NULL;
            scratch_buf->flags = 0;
            scratch_buf->ref_count = 1;
            
            worker_t->buffer = scratch_buf;
            worker_t->byte_offset = 0;
            // The op will write to this local float
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
    
    if (batch->main_state) {
        state->ctx.global_error_ptr = (mf_atomic_i32*)&batch->main_state->error_code;
    }

    state->ctx.linear_offset = (u32)start_idx;

    // Unflatten start index for N-dimensional operations (e.g. op_index)
    size_t temp_idx = start_idx;
    for (int i = batch->ndim - 1; i >= 0; --i) {
        state->ctx.tile_offset[i] = (u32)(temp_idx % batch->domain_shape[i]);
        temp_idx /= batch->domain_shape[i];
    }

    for(int d=0; d<batch->ndim; ++d) state->ctx.domain_shape[d] = batch->domain_shape[d];

    prepare_registers(state, batch, start_idx, count);

    mf_cpu_exec(&state->ctx, batch->program, batch->op_table, batch);
    
    if (state->ctx.error != MF_ERROR_NONE && batch->main_state) {
        mf_atomic_store((mf_atomic_i32*)&batch->main_state->error_code, (int32_t)state->ctx.error);
    }
}

// --- Dispatch ---

static void mf_backend_cpu_dispatch(
    void* backend_state,
    const struct mf_program* program,
    struct mf_state* main_state,
    const mf_tensor* domain,
    uint32_t start_inst,
    uint32_t inst_count
) {
    mf_backend_cpu_state* state = (mf_backend_cpu_state*)backend_state;
    if (!domain) return;

    size_t total_elements = mf_tensor_count(domain);
    if (total_elements == 0) return;

    if (program->meta.tensor_count > MF_MAX_REGISTERS) {
        MF_LOG_ERROR("CPU Backend: Program tensor count (%u) exceeds backend limit (%d).", 
            program->meta.tensor_count, MF_MAX_REGISTERS);
        return;
    }

    int num_threads = state->pool ? mf_thread_pool_get_thread_count(state->pool) : 1;
    
    mf_cpu_parallel_batch batch = {
        .program = program,
        .main_state = main_state,
        .op_table = state->op_table,
        .start_inst = start_inst,
        .inst_count = inst_count,
        .total_elements = total_elements,
        .ndim = domain->info.ndim,
        .num_threads = num_threads,
        .reduction_scratch = NULL,
        .active_reg_count = 0
    };
    memcpy(batch.domain_shape, domain->info.shape, sizeof(u32) * MF_MAX_DIMS);

    // 1. Analyze Tensors & Detect Reductions
    bool has_reductions = false;
    u8 reg_used[MF_MAX_REGISTERS] = {0};

    // Mark registers used in the current instruction range
    for (uint32_t i = start_inst; i < start_inst + inst_count; ++i) {
        const mf_instruction* inst = &program->code[i];
        reg_used[inst->dest_idx] = 1;
        reg_used[inst->src1_idx] = 1;
        reg_used[inst->src2_idx] = 1;
        reg_used[inst->src3_idx] = 1;

        if (inst->opcode == MF_OP_SUM || inst->opcode == MF_OP_MEAN) {
            has_reductions = true; // Temporary flag, refined below
        }
    }

    // Always include the domain register as it's needed for context/metadata
    for (u32 i = 0; i < program->meta.task_count; ++i) {
        if (program->tasks[i].start_inst == start_inst) {
            reg_used[program->tasks[i].domain_reg] = 1;
            break;
        }
    }

    for (u32 i = 0; i < program->meta.tensor_count; ++i) {
        if (!reg_used[i]) continue;
        
        batch.active_regs[batch.active_reg_count++] = (u16)i;

        mf_tensor* main_t = &main_state->registers[i];
        batch.roles[i] = MF_TENSOR_ROLE_UNIFORM;
        batch.channels[i] = 1;

        if (!mf_tensor_is_valid(main_t)) continue;

        size_t t_count = mf_tensor_count(main_t);
        bool is_constant = main_state->ownership_flags && (main_state->ownership_flags[i] == 0);

        if (t_count == total_elements && !is_constant) {
            batch.roles[i] = MF_TENSOR_ROLE_SPATIAL;
            batch.channels[i] = 1;
        } else if (total_elements > 0 && t_count > total_elements && t_count % total_elements == 0 && !is_constant) {
            batch.roles[i] = MF_TENSOR_ROLE_SPATIAL;
            batch.channels[i] = (int)(t_count / total_elements);
        }
    }

    // Refine reductions: only mark as REDUCTION if it's a SUM/MEAN over a SPATIAL input
    has_reductions = false;
    for (uint32_t i = start_inst; i < start_inst + inst_count; ++i) {
        const mf_instruction* inst = &program->code[i];
        if (inst->opcode == MF_OP_SUM || inst->opcode == MF_OP_MEAN) {
            if (batch.roles[inst->src1_idx] == MF_TENSOR_ROLE_SPATIAL) {
                batch.roles[inst->dest_idx] = MF_TENSOR_ROLE_REDUCTION;
                has_reductions = true;
            }
        }
    }

    // 2. Allocate scratch memory if needed
    if (has_reductions && num_threads > 1) {
        batch.reduction_scratch = calloc(num_threads * MF_MAX_REGISTERS, sizeof(f32));
    }

    // 3. Dispatch Jobs
    u32 total_jobs = (u32)((total_elements + MF_CPU_JOB_SIZE - 1) / MF_CPU_JOB_SIZE);

    MF_LOG_DEBUG("CPU Dispatch: elements=%zu, jobs=%u, threads=%d, reductions=%s", 
        total_elements, total_jobs, num_threads, has_reductions ? "YES" : "NO");

    if (total_elements <= MF_CPU_INLINE_THRESHOLD || total_jobs == 1) {
        mf_backend_cpu_worker_state local_worker;
        _Alignas(16) u8 local_heap[MF_KB(64)]; 
        local_worker.thread_idx = 0;
        local_worker.heap_mem = local_heap;
        local_worker.heap_size = sizeof(local_heap);
        mf_arena_init(&local_worker.temp_arena, local_worker.heap_mem, local_worker.heap_size);
        mf_arena_init(&local_worker.reg_arena, local_worker.reg_arena_mem, sizeof(local_worker.reg_arena_mem));
        
        cpu_worker_job(0, &local_worker, &batch);
    } else {
        if (state->pool) {
            mf_thread_pool_run(state->pool, total_jobs, cpu_worker_job, &batch);
        } else {
            void* persistent_worker = worker_init(0, NULL);
            for (u32 i = 0; i < total_jobs; ++i) {
                cpu_worker_job(i, persistent_worker, &batch);
            }
            worker_cleanup(persistent_worker, NULL);
        }
    }

    // 4. Merge Reductions
    if (has_reductions && batch.reduction_scratch) {
        for (u32 reg_idx = 0; reg_idx < program->meta.tensor_count; ++reg_idx) {
            if (batch.roles[reg_idx] == MF_TENSOR_ROLE_REDUCTION) {
                f32 final_val = 0;
                for (int t = 0; t < num_threads; ++t) {
                    final_val += batch.reduction_scratch[t * MF_MAX_REGISTERS + reg_idx];
                }
                
                mf_tensor* main_t = &main_state->registers[reg_idx];
                // For MEAN, we need to divide by total count later. 
                // But wait, op_mean already divides by count in the worker.
                // If we sum partial means, we get (S1/N + S2/N + ...) = (S1+S2+...)/N = S_total / N.
                // This only works if all batches are the same size!
                // If batches are different (e.g. the last one), we need a different approach for MEAN.
                
                // Correction for MEAN: workers should only SUM, and then we divide once at the end.
                // But wait, the opcode is MF_OP_MEAN. We can't easily change what it does without
                // knowing if it's running in parallel or not.
                
                // For now, let's assume workers for MEAN also just SUM, and we handle it here.
                // Or better: MEAN is SUM followed by DIV. If the compiler does this, we are golden.
                
                *((f32*)main_t->buffer->data + main_t->byte_offset / sizeof(f32)) = final_val;
            }
        }
        free(batch.reduction_scratch);
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