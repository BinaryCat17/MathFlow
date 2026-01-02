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

typedef enum {
    MF_SRC_BUFFER,    // Data is in a global buffer (Resource or Constant)
    MF_SRC_GENERATOR, // Data needs to be generated (Builtin like host.index)
    MF_SRC_SCRATCH    // Temporary buffer (Scratchpad)
} mf_reg_source_type;

typedef struct {
    mf_reg_source_type type;
    
    // For BUFFER
    mf_buffer* buffer;
    size_t base_offset;
    i8 stride;

    // For GENERATOR
    mf_builtin_id builtin_id;
    u8 builtin_axis;
      
    mf_type_info info;     
} mf_cpu_reg_plan;

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
    mf_cpu_reg_plan plans[MF_MAX_REGISTERS];

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

static const char* _dtype_to_str(mf_dtype type) {
    switch(type) {
        case MF_DTYPE_F32: return "F32";
        case MF_DTYPE_I32: return "I32";
        case MF_DTYPE_U8:  return "U8";
        default: return "UNK";
    }
}

static const char* find_reg_name(const mf_program* prog, u32 reg_idx) {
    if (!prog || !prog->symbols) return "temp";
    for (u32 i = 0; i < prog->meta.symbol_count; ++i) {
        if (prog->symbols[i].register_idx == reg_idx) return prog->symbols[i].name;
    }
    return "temp";
}

static void format_tensor_debug(char* buf, const mf_tensor* t, int reg_idx, const mf_program* prog) {
    if (!t) {
        sprintf(buf, "Reg %-2d (NULL)", reg_idx);
        return;
    }
    
    const char* name = find_reg_name(prog, reg_idx);
    char shape_str[64] = {0};
    int pos = 0;
    if (t->info.ndim == 0) {
        strcpy(shape_str, "Scalar");
    } else {
        for (int i = 0; i < t->info.ndim; ++i) {
            pos += sprintf(shape_str + pos, "%d%s", t->info.shape[i], (i < t->info.ndim - 1) ? "," : "");
        }
    }

    char tag[128];
    sprintf(tag, "Reg %-2d (%s)", reg_idx, name);

    if (!t->buffer || !t->buffer->data) {
        sprintf(buf, "% -20s : <Unallocated> [%s] Shape: [%s]", tag, _dtype_to_str(t->info.dtype), shape_str);
        return;
    }

    if (t->info.ndim == 0 || (t->info.ndim == 1 && t->info.shape[0] == 1)) {
        void* d = (u8*)t->buffer->data + t->byte_offset;
        float val = 0;
        if (t->info.dtype == MF_DTYPE_F32) val = *(f32*)d;
        else if (t->info.dtype == MF_DTYPE_I32) val = (f32)*(int32_t*)d;
        else if (t->info.dtype == MF_DTYPE_U8) val = (f32)*(u8*)d;
        sprintf(buf, "% -20s : Value: %-10.3f (%s)", tag, val, _dtype_to_str(t->info.dtype));
    } else {
        sprintf(buf, "% -20s : Tensor[%-10s] (%s) Ptr: %p", tag, shape_str, _dtype_to_str(t->info.dtype), (void*)((u8*)t->buffer->data + t->byte_offset));
    }
}

static void report_crash(mf_exec_ctx* ctx, const mf_cpu_parallel_batch* batch, u32 inst_idx) {
    const mf_instruction* inst = &batch->program->code[inst_idx];

    char coords[128] = {0};
    int pos = 0;
    
    // Calculate exact coordinates of the failing element
    u32 exact_linear = ctx->linear_offset + ctx->error_idx;
    u32 temp_idx = exact_linear;
    u32 exact_coords[MF_MAX_DIMS];
    for (int i = ctx->ndim - 1; i >= 0; --i) {
        exact_coords[i] = temp_idx % ctx->domain_shape[i];
        temp_idx /= ctx->domain_shape[i];
    }

    for (int d = 0; d < ctx->ndim; ++d) {
        pos += sprintf(coords + pos, "%u%s", exact_coords[d], (d < ctx->ndim - 1) ? ", " : "");
    }

    char s1_info[128], s2_info[128], s3_info[128], s4_info[128], d_info[128];
    u32 reg_count = batch->program->meta.tensor_count;
    format_tensor_debug(d_info,  &ctx->registers[inst->dest_idx], inst->dest_idx, batch->program);
    format_tensor_debug(s1_info, (inst->src1_idx < reg_count) ? &ctx->registers[inst->src1_idx] : NULL, inst->src1_idx, batch->program);
    format_tensor_debug(s2_info, (inst->src2_idx < reg_count) ? &ctx->registers[inst->src2_idx] : NULL, inst->src2_idx, batch->program);
    format_tensor_debug(s3_info, (inst->src3_idx < reg_count) ? &ctx->registers[inst->src3_idx] : NULL, inst->src3_idx, batch->program);
    format_tensor_debug(s4_info, (inst->src4_idx < reg_count) ? &ctx->registers[inst->src4_idx] : NULL, inst->src4_idx, batch->program);

    MF_LOG_FATAL("\n"
                 "================================================================================\n"
                 "                             KERNEL CRASH REPORT\n"
                 "================================================================================\n"
                 "  FAILED INSTRUCTION:\n"
                 "  #%u Opcode: %s [%d]\n"
                 "\n"
                 "  OPERANDS:\n"
                 "  Dest: %s\n"
                 "  Src1: %s\n"
                 "  Src2: %s\n"
                 "  Src3: %s\n"
                 "  Src4: %s\n"
                 "\n"
                 "  EXECUTION CONTEXT:\n"
                 "  Domain Coord : [%s]\n"
                 "  Linear Index : %u (Batch Offset: %u)\n"
                 "  Error Type   : %s\n"
                 "================================================================================\n",
                 inst_idx, mf_opcode_to_str(inst->opcode), inst->opcode, 
                 d_info, s1_info, s2_info, s3_info, s4_info,
                 coords, exact_linear, ctx->error_idx,
                 mf_exec_error_to_str(ctx->error));
}

static inline void mf_cpu_exec(mf_exec_ctx* ctx, const mf_cpu_parallel_batch* batch, u32 count) {
    for (uint32_t i = 0; i < count; ++i) {
        // Stop if local error OR global error detected by another thread
        if (ctx->error != MF_ERROR_NONE) break;
        if (batch->main_state && mf_atomic_load((mf_atomic_i32*)&batch->main_state->error_code) != 0) break;
        if (ctx->global_error_ptr && mf_atomic_load(ctx->global_error_ptr) != 0) break;

        u32 inst_idx = batch->start_inst + i;
        const mf_instruction* inst = &batch->program->code[inst_idx];
        mf_op_func op = batch->op_table[inst->opcode];
        if (op) {
            op(ctx, inst);
            
            if (ctx->error != MF_ERROR_NONE) {
                report_crash(ctx, batch, inst_idx);
                break;
            }
        }
    }
}

static void mf_generate_index_chunk(f32* out, u32 count, u32 job_offset, u8 axis, bool is_vector, u8 domain_ndim, const u32* domain_shape) {
    u32 current_coords[MF_MAX_DIMS];
    u32 temp_idx = job_offset;
    for (int i = domain_ndim - 1; i >= 0; --i) {
        current_coords[i] = temp_idx % domain_shape[i];
        temp_idx /= domain_shape[i];
    }

    u32 vec_size = is_vector ? domain_ndim : 1;

    for (u32 e = 0; e < count; ++e) {
        if (is_vector) {
            for (u32 d = 0; d < domain_ndim; ++d) {
                out[e * domain_ndim + d] = (f32)current_coords[d];
            }
        } else {
            out[e] = (axis < domain_ndim) ? (f32)current_coords[axis] : 0.0f;
        }
        
        // Advance coords
        for (int d = (int)domain_ndim - 1; d >= 0; --d) {
            current_coords[d]++;
            if (current_coords[d] < domain_shape[d] || d == 0) break;
            current_coords[d] = 0;
        }
    }
}

static void prepare_registers(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch, size_t start_idx, size_t count) {
    mf_exec_ctx* ctx = &state->ctx;
    int tid = state->thread_idx;
    
    for (size_t k = 0; k < batch->active_reg_count; ++k) {
        u16 i = batch->active_regs[k];
        const mf_cpu_reg_plan* plan = &batch->plans[i];
        mf_tensor* t = &ctx->registers[i];

        t->info = plan->info;

        switch (plan->type) {
            case MF_SRC_BUFFER: {
                t->buffer = plan->buffer;
                size_t dtype_sz = mf_dtype_size(t->info.dtype);
                t->byte_offset = plan->base_offset + (start_idx * (size_t)plan->stride * dtype_sz);
                
                // Adjust metadata for the flat window view
                t->info.ndim = (plan->stride > 1) ? 2 : 1;
                t->info.shape[0] = (int32_t)count;
                t->info.strides[0] = plan->stride;
                if (plan->stride > 1) {
                    t->info.shape[1] = plan->stride;
                    t->info.strides[1] = 1;
                } else if (plan->stride == 0) {
                    t->info.strides[0] = 0;
                }
            } break;

            case MF_SRC_GENERATOR: {
                if (plan->builtin_id == MF_BUILTIN_INDEX) {
                    bool is_vector = (t->info.ndim > 0 && t->info.shape[t->info.ndim-1] > 1 && t->info.ndim > 1);
                    // Special case: if ndim=1 and shape[0] > 1, it might be a vector of indices if it's the only dim.
                    // But usually host.index.N is a scalar stream (stride 1).
                    // If it's just "host.index", it's a vector stream.
                    
                    // The compiler should have set the shape correctly. 
                    // If it's a vector, shape[1] will be domain_ndim.
                    is_vector = (t->info.ndim == 2); 
                    
                    size_t vec_size = is_vector ? batch->ndim : 1;
                    size_t bytes = count * vec_size * sizeof(f32);
                    void* mem = mf_exec_ctx_scratch_alloc(ctx, bytes);
                    if (mem) {
                        mf_generate_index_chunk((f32*)mem, (u32)count, (u32)start_idx, plan->builtin_axis, is_vector, batch->ndim, batch->domain_shape);
                        
                        mf_buffer* scratch_buf = MF_ARENA_PUSH(&state->reg_arena, mf_buffer, 1);
                        mf_buffer_init_view(scratch_buf, mem, bytes);
                        t->buffer = scratch_buf;
                        t->byte_offset = 0;
                        
                        t->info.ndim = (uint8_t)(is_vector ? 2 : 1);
                        t->info.shape[0] = (int32_t)count;
                        t->info.strides[0] = plan->stride;
                        if (is_vector) {
                            t->info.shape[1] = (int32_t)batch->ndim;
                            t->info.strides[1] = 1;
                        }
                    }
                } else {
                    // host.time, resolution, etc. are currently treated as constants by the engine
                    // but here we could handle them as streams if needed.
                    // For now, if they are MF_SRC_GENERATOR, they should have been handled by engine binding.
                    // If we reach here, it's a fallback or not implemented.
                    MF_LOG_WARN("CPU Backend: Builtin %d not fully implemented in worker.", plan->builtin_id);
                }
            } break;

            case MF_SRC_SCRATCH: {
                size_t elements = count * (plan->stride > 0 ? (size_t)plan->stride : 1); 
                size_t dt_size = mf_dtype_size(t->info.dtype);
                size_t bytes = elements * dt_size;
                
                void* mem = mf_exec_ctx_scratch_alloc(ctx, bytes);
                if (mem) {
                    mf_buffer* scratch_buf = MF_ARENA_PUSH(&state->reg_arena, mf_buffer, 1);
                    mf_buffer_init_view(scratch_buf, mem, bytes);
                    t->buffer = scratch_buf;
                    t->byte_offset = 0;
                    
                    t->info.ndim = (plan->stride > 1) ? 2 : 1;
                    t->info.shape[0] = (int32_t)count;
                    t->info.strides[0] = plan->stride;
                    if (plan->stride > 1) {
                        t->info.shape[1] = plan->stride;
                        t->info.strides[1] = 1;
                    } else if (plan->stride == 0) {
                        t->info.strides[0] = 0;
                    }
                }
            } break;
        }

        // Reductions still need special handling
        if (batch->reduction_scratch && plan->stride == -1) {
            mf_buffer* scratch_buf = MF_ARENA_PUSH(&state->reg_arena, mf_buffer, 1);
            scratch_buf->data = &batch->reduction_scratch[tid * MF_MAX_REGISTERS + i];
            scratch_buf->size_bytes = sizeof(f32);
            scratch_buf->alloc = NULL;
            scratch_buf->flags = 0;
            scratch_buf->ref_count = 1;
            
            t->buffer = scratch_buf;
            t->byte_offset = 0;
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

    mf_cpu_exec(&state->ctx, batch, batch->inst_count);
    
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

    // 1. Build Execution Plan
    u8 reg_processed[MF_MAX_REGISTERS] = {0};
    bool has_reductions = false;

    for (uint32_t i = start_inst; i < start_inst + inst_count; ++i) {
        const mf_instruction* inst = &program->code[i];
        
        uint16_t regs[] = { inst->dest_idx, inst->src1_idx, inst->src2_idx, inst->src3_idx, inst->src4_idx };
        int8_t reg_strides[] = { inst->strides[0], inst->strides[1], inst->strides[2], inst->strides[3], 0 };

        for (int r = 0; r < 5; ++r) {
            uint16_t reg_idx = regs[r];
            if (reg_idx >= program->meta.tensor_count) continue;

            if (!reg_processed[reg_idx]) {
                mf_cpu_reg_plan* plan = &batch.plans[reg_idx];
                mf_tensor* main_t = &main_state->registers[reg_idx];
                plan->info = main_t->info;
                plan->stride = reg_strides[r];
                
                if (plan->stride == -1) has_reductions = true;

                // Check for Builtin ID from symbols
                mf_bin_symbol* sym = NULL;
                for (u32 s = 0; s < program->meta.symbol_count; ++s) {
                    if (program->symbols[s].register_idx == reg_idx) {
                        sym = &program->symbols[s];
                        break;
                    }
                }

                if (sym && sym->builtin_id != MF_BUILTIN_NONE) {
                    plan->type = MF_SRC_GENERATOR;
                    plan->builtin_id = (mf_builtin_id)sym->builtin_id;
                    plan->builtin_axis = sym->builtin_axis;
                } else if (main_t->buffer) {
                    plan->type = MF_SRC_BUFFER;
                    plan->buffer = main_t->buffer;
                    plan->base_offset = main_t->byte_offset;
                } else {
                    plan->type = MF_SRC_SCRATCH;
                }

                // Safety check: if total elements in register is less than the execution domain,
                // it MUST be a broadcast (stride 0) or we will crash.
                if (mf_tensor_count(main_t) < total_elements && plan->stride > 0) {
                    plan->stride = 0;
                }

                batch.active_regs[batch.active_reg_count++] = reg_idx;
                reg_processed[reg_idx] = 1;
            } else if (inst->opcode != MF_OP_NOOP) {
                // If register already processed, update stride if the current one is more "spatial"
                // This handles cases where a register is used as both constant and spatial in different ops
                // (though compiler should ideally handle this)
                if (reg_strides[r] != 0 && batch.plans[reg_idx].stride == 0) {
                    batch.plans[reg_idx].stride = reg_strides[r];
                }
            }
        }
    }

    // 2. Allocate scratch memory if needed
    if (has_reductions && num_threads > 1) {
        batch.reduction_scratch = calloc(num_threads * MF_MAX_REGISTERS, sizeof(f32));
    }

    // 3. Dispatch Jobs
    u32 total_jobs = (u32)((total_elements + MF_CPU_JOB_SIZE - 1) / MF_CPU_JOB_SIZE);

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
            if (reg_processed[reg_idx] && batch.plans[reg_idx].stride == -1) {
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