#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_state.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_thread_pool.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_platform.h>
#include <mathflow/base/mf_shape.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdalign.h>

// --- Constants ---

#define MF_CPU_JOB_SIZE         4096         // Elements per job (Linear)
#define MF_CPU_INLINE_THRESHOLD 1024         // If total elements < this, run inline
#define MF_CPU_WORKER_HEAP_SZ   (64*1024*1024) // 64MB per worker
#define MF_CPU_REG_ARENA_SZ     (128*1024)    // 128KB for registers metadata
#define MF_MAX_REGISTERS        512          // Max tensors per program

// --- Internal Structures ---

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
    i32 strides[MF_MAX_REGISTERS];

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

static void report_crash(mf_exec_ctx* ctx, const mf_instruction* inst, uint32_t inst_idx) {
    char coords[128] = {0};
    int pos = 0;
    for (int d = 0; d < ctx->ndim; ++d) {
        pos += sprintf(coords + pos, "%u%s", ctx->tile_offset[d], (d < ctx->ndim - 1) ? ", " : "");
    }

    MF_LOG_FATAL("\n"
                 "==================================================\n"
                 "             KERNEL CRASH REPORT\n"
                 "==================================================\n"
                 "  Instruction : #%u (Opcode: %s [%d])\n"
                 "  Registers   : D:%u, S1:%u, S2:%u, S3:%u, S4:%u\n"
                 "  Domain Coord: [%s]\n"
                 "  Linear Index: %u\n"
                 "  Error Type  : %s\n"
                 "==================================================",
                 inst_idx, mf_opcode_to_str(inst->opcode), inst->opcode, 
                 inst->dest_idx, inst->src1_idx, inst->src2_idx, inst->src3_idx, inst->src4_idx,
                 coords, ctx->linear_offset,
                 mf_exec_error_to_str(ctx->error));
}

static f32 get_val(mf_tensor* t) {
    if (!t || !t->buffer || !t->buffer->data) return 0;
    void* d = (u8*)t->buffer->data + t->byte_offset;
    if (t->info.dtype == MF_DTYPE_F32) return *(f32*)d;
    if (t->info.dtype == MF_DTYPE_I32) return (f32)*(int32_t*)d;
    if (t->info.dtype == MF_DTYPE_U8) return (f32)*(u8*)d;
    return 0;
}

static inline void mf_cpu_exec(mf_exec_ctx* ctx, const mf_program* program, mf_op_func* op_table, const mf_cpu_parallel_batch* batch) {
    const mf_instruction* code = program->code;
    const uint32_t end_inst = batch->start_inst + batch->inst_count;
    
    for (uint32_t i = batch->start_inst; i < end_inst; ++i) {
        // Stop if local error OR global error detected by another thread
        if (ctx->error != MF_ERROR_NONE) break;
        if (batch->main_state && mf_atomic_load((mf_atomic_i32*)&batch->main_state->error_code) != 0) break;
        if (ctx->global_error_ptr && mf_atomic_load(ctx->global_error_ptr) != 0) break;

        const mf_instruction* inst = &code[i];
        
        if (ctx->linear_offset == 0) {
            f32 v_s1 = 0, v_s2 = 0, v_s3 = 0, v_s4 = 0;
            mf_tensor* t1 = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
            mf_tensor* t2 = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
            mf_tensor* t3 = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ);
            mf_tensor* t4 = mf_exec_ctx_map_tensor(ctx, inst->src4_idx, MF_ACCESS_READ);
            if (t1) v_s1 = get_val(t1);
            if (t2) v_s2 = get_val(t2);
            if (t3) v_s3 = get_val(t3);
            if (t4) v_s4 = get_val(t4);
            MF_LOG_DEBUG("Exec #%u: %s (D:%u, S1:%u[%.2f], S2:%u[%.2f], S3:%u[%.2f], S4:%u[%.2f]) Strides: [%d, %d, %d, %d, %d]", 
                i, mf_opcode_to_str(inst->opcode), inst->dest_idx, 
                inst->src1_idx, v_s1, inst->src2_idx, v_s2, inst->src3_idx, v_s3, inst->src4_idx, v_s4,
                inst->strides[0], inst->strides[1], inst->strides[2], inst->strides[3], inst->strides[4]);
        }

        mf_op_func op = op_table[inst->opcode];
        if (op) {
            op(ctx, inst);
            
            if (ctx->error != MF_ERROR_NONE) {
                report_crash(ctx, inst, i);
                break;
            }
        }
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

        i32 stride = batch->strides[i];
        if (main_t->buffer) {
            size_t dtype_sz = mf_dtype_size(main_t->info.dtype);
            worker_t->byte_offset = main_t->byte_offset + (start_idx * (size_t)stride * dtype_sz);
            
            if (i == 57 || stride > 1) {
                MF_LOG_TRACE("  Prep Reg %u: External (Buffer: %p, Size: %zu, Stride: %d, Offset: %zu)", 
                    i, (void*)main_t->buffer->data, main_t->buffer->size_bytes, stride, worker_t->byte_offset);
            }
        } else {
            // This is a temporary register (scratchpad). Allocate memory in worker's arena.
            size_t elements = count * (stride > 0 ? (size_t)stride : 1); 
            size_t dt_size = mf_dtype_size(main_t->info.dtype);
            if (dt_size == 0) dt_size = 4; // Fallback to float size if unknown
            size_t bytes = elements * dt_size;
            
            void* mem = mf_exec_ctx_scratch_alloc(worker_ctx, bytes);
            if (mem) {
                mf_buffer* scratch_buf = MF_ARENA_PUSH(&state->reg_arena, mf_buffer, 1);
                mf_buffer_init_view(scratch_buf, mem, bytes);
                worker_t->buffer = scratch_buf;
                worker_t->byte_offset = 0;
                
                if (stride > 1) {
                    MF_LOG_TRACE("  Prep Reg %u: Scratchpad (Size: %zu, Stride: %d)", i, bytes, stride);
                }
            } else {
                MF_LOG_FATAL("CPU Backend: Out of scratchpad memory for register %u", i);
            }
        }
        
        // Adjust metadata for the flat window view
        if (stride > 0) {
            worker_t->info.ndim = (stride > 1) ? 2 : 1;
            worker_t->info.shape[0] = (int32_t)count;
            worker_t->info.strides[0] = stride;
            if (stride > 1) {
                worker_t->info.shape[1] = stride;
                worker_t->info.strides[1] = 1;
            }
        } else if (batch->reduction_scratch && stride == -1) {
            // Redirect output to thread-local scratch space
            mf_buffer* scratch_buf = MF_ARENA_PUSH(&state->reg_arena, mf_buffer, 1);
            scratch_buf->data = &batch->reduction_scratch[tid * MF_MAX_REGISTERS + i];
            scratch_buf->size_bytes = sizeof(f32);
            scratch_buf->alloc = NULL;
            scratch_buf->flags = 0;
            scratch_buf->ref_count = 1;
            
            worker_t->buffer = scratch_buf;
            worker_t->byte_offset = 0;
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
        state->ctx.global_error_ptr = batch->main_state->global_error_ptr ? 
                                      batch->main_state->global_error_ptr : 
                                      &batch->main_state->error_code;
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
        mf_atomic_store(&batch->main_state->error_code, (int32_t)state->ctx.error);
        if (batch->main_state->global_error_ptr) {
            mf_atomic_store(batch->main_state->global_error_ptr, (int32_t)state->ctx.error);
        }
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
    if (total_elements == 0) {
        MF_LOG_WARN("CPU Backend: Dispatch ignored. Domain has 0 elements.");
        return;
    }

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
    u8 reg_processed[MF_MAX_REGISTERS] = {0};
    memset(batch.strides, 0, sizeof(batch.strides));

    // Initialize strides from instructions in the current range
    for (uint32_t i = start_inst; i < start_inst + inst_count; ++i) {
        const mf_instruction* inst = &program->code[i];
        
        uint16_t regs[] = { inst->dest_idx, inst->src1_idx, inst->src2_idx, inst->src3_idx, inst->src4_idx };
        for (int r = 0; r < 5; ++r) {
            uint16_t reg_idx = regs[r];
            if (!reg_processed[reg_idx]) {
                batch.strides[reg_idx] = inst->strides[r];
                batch.active_regs[batch.active_reg_count++] = reg_idx;
                reg_processed[reg_idx] = 1;
            } else {
                // If register is used multiple times, ensure we take the non-zero stride 
                // (e.g. if used as both uniform and spatial)
                if (inst->strides[r] > batch.strides[reg_idx]) {
                    batch.strides[reg_idx] = inst->strides[r];
                }
            }
        }
    }

    // Refine reductions: only mark as REDUCTION (-1) if it's a SUM over a SPATIAL input
    bool has_reductions = false;
    for (uint32_t i = start_inst; i < start_inst + inst_count; ++i) {
        const mf_instruction* inst = &program->code[i];
        if (inst->opcode == MF_OP_SUM) {
            if (batch.strides[inst->src1_idx] > 0) {
                batch.strides[inst->dest_idx] = -1; // Flag for reduction
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
        _Alignas(16) u8 local_heap[MF_MB(4)]; 
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
            if (batch.strides[reg_idx] == -1) {
                f32 final_val = 0;
                for (int t = 0; t < num_threads; ++t) {
                    final_val += batch.reduction_scratch[t * MF_MAX_REGISTERS + reg_idx];
                }
                
                mf_tensor* main_t = &main_state->registers[reg_idx];
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
