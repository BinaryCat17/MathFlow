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
    ssize_t stride_bytes;

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

    // Parallel Sync Support
    int sync_pass;
    void* sync_data;

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

static void format_tensor_debug(char* buf, const mf_exec_ctx* ctx, int reg_idx, const mf_program* prog, const char* port_name) {
    if (reg_idx < 0 || reg_idx >= MF_MAX_REGISTERS) {
        sprintf(buf, "Reg %-2d (INVALID)", reg_idx);
        return;
    }
    
    const char* name = find_reg_name(prog, reg_idx);
    const mf_type_info* info = &ctx->reg_info[reg_idx];
    void* data = ctx->reg_ptrs[reg_idx];

    char shape_str[64] = {0};
    int pos = 0;
    if (info->ndim == 0) {
        strcpy(shape_str, "Scalar");
    } else {
        for (int i = 0; i < info->ndim; ++i) {
            pos += sprintf(shape_str + pos, "%d%s", info->shape[i], (i < info->ndim - 1) ? "," : "");
        }
    }

    char tag[128];
    if (port_name) {
        sprintf(tag, "Reg %-2d (%s) [%s]", reg_idx, name, port_name);
    } else {
        sprintf(tag, "Reg %-2d (%s)", reg_idx, name);
    }

    if (!data) {
        sprintf(buf, "%-30s : <NULL PTR> [%s] Shape: [%s]", tag, _dtype_to_str(info->dtype), shape_str);
        return;
    }

    if (info->ndim == 0 || (info->ndim == 1 && info->shape[0] == 1)) {
        float val = 0;
        if (info->dtype == MF_DTYPE_F32) val = *(f32*)data;
        else if (info->dtype == MF_DTYPE_I32) val = (f32)*(int32_t*)data;
        else if (info->dtype == MF_DTYPE_U8) val = (f32)*(u8*)data;
        sprintf(buf, "%-30s : Value: %-10.3f (%s)", tag, val, _dtype_to_str(info->dtype));
    } else {
        sprintf(buf, "%-30s : Tensor[%-10s] (%s) Ptr: %p", tag, shape_str, _dtype_to_str(info->dtype), data);
    }
}

static void report_crash(mf_exec_ctx* ctx, const mf_cpu_parallel_batch* batch, u32 inst_idx) {
    const mf_instruction* inst = &batch->program->code[inst_idx];
    const mf_runtime_op_metadata* meta = mf_get_op_metadata(inst->opcode);

    char coords[128] = {0};
    int pos = 0;
    
    // Calculate exact coordinates of the failing element
    u32 exact_linear = ctx->linear_offset + ctx->error_idx;
    u32 temp_idx = exact_linear;
    u32 exact_coords[MF_MAX_DIMS];
    for (int i = (int)ctx->ndim - 1; i >= 0; --i) {
        exact_coords[i] = temp_idx % ctx->domain_shape[i];
        temp_idx /= ctx->domain_shape[i];
    }

    for (int d = 0; d < ctx->ndim; ++d) {
        pos += sprintf(coords + pos, "%u%s", exact_coords[d], (d < ctx->ndim - 1) ? ", " : "");
    }

    char s1_info[128], s2_info[128], s3_info[128], s4_info[128], d_info[128];
    u32 reg_count = batch->program->meta.tensor_count;

    format_tensor_debug(d_info,  ctx, inst->dest_idx, batch->program, "out");
    format_tensor_debug(s1_info, ctx, inst->src1_idx, batch->program, meta ? meta->ports[0] : "src1");
    format_tensor_debug(s2_info, ctx, inst->src2_idx, batch->program, meta ? meta->ports[1] : "src2");
    format_tensor_debug(s3_info, ctx, inst->src3_idx, batch->program, meta ? meta->ports[2] : "src3");
    format_tensor_debug(s4_info, ctx, inst->src4_idx, batch->program, meta ? meta->ports[3] : "src4");

    MF_LOG_FATAL("\n"
                 "================================================================================\n"
                 "                             KERNEL CRASH REPORT\n"
                 "================================================================================\n"
                 "  FAILED INSTRUCTION:\n"
                 "  #%u Opcode: %s [%d] at line %u, col %u\n"
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
                 inst_idx, mf_opcode_to_str(inst->opcode), inst->opcode, inst->line, inst->column,
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

static void mf_generate_index_chunk(void* out_raw, mf_dtype dtype, u32 count, u32 job_offset, u8 axis, bool is_vector, u8 domain_ndim, const u32* domain_shape) {
    u32 current_coords[MF_MAX_DIMS];
    u32 temp_idx = job_offset;
    for (int i = domain_ndim - 1; i >= 0; --i) {
        current_coords[i] = temp_idx % domain_shape[i];
        temp_idx /= domain_shape[i];
    }

    for (u32 e = 0; e < count; ++e) {
        if (is_vector) {
            for (u32 d = 0; d < domain_ndim; ++d) {
                float val = (f32)current_coords[d];
                if (dtype == MF_DTYPE_F32) ((f32*)out_raw)[e * domain_ndim + d] = val;
                else if (dtype == MF_DTYPE_I32) ((i32*)out_raw)[e * domain_ndim + d] = (i32)current_coords[d];
            }
        } else {
            float val = (axis < domain_ndim) ? (f32)current_coords[axis] : 0.0f;
            if (dtype == MF_DTYPE_F32) ((f32*)out_raw)[e] = val;
            else if (dtype == MF_DTYPE_I32) ((i32*)out_raw)[e] = (axis < domain_ndim) ? (i32)current_coords[axis] : 0;
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
    const mf_program* prog = batch->program;
    int tid = state->thread_idx;
    
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        const mf_cpu_reg_plan* plan = &batch->plans[i];
        ctx->reg_info[i] = plan->info;

        switch (plan->type) {
            case MF_SRC_BUFFER:
                ctx->reg_ptrs[i] = (u8*)plan->buffer->data + plan->base_offset + (start_idx * plan->stride_bytes);
                break;

            case MF_SRC_GENERATOR:
                if (plan->builtin_id == MF_BUILTIN_INDEX) {
                    bool is_vector = (plan->info.ndim > batch->ndim);
                    size_t vec_size = is_vector ? (size_t)plan->info.shape[plan->info.ndim - 1] : 1;
                    size_t bytes = count * vec_size * mf_dtype_size(plan->info.dtype);
                    void* mem = mf_exec_ctx_scratch_alloc(ctx, bytes);
                    if (mem) {
                        mf_generate_index_chunk(mem, plan->info.dtype, (u32)count, (u32)start_idx, plan->builtin_axis, is_vector, batch->ndim, batch->domain_shape);
                        ctx->reg_ptrs[i] = mem;
                    }
                }
                break;

            case MF_SRC_SCRATCH: {
                size_t dtype_sz = mf_dtype_size(plan->info.dtype);
                // For scratch, if stride is 0 (broadcast/scalar), we only need 1 element per job
                size_t elements = (plan->stride_bytes != 0) ? count : 1;
                size_t bytes = elements * dtype_sz;
                void* mem = mf_exec_ctx_scratch_alloc(ctx, bytes);
                ctx->reg_ptrs[i] = mem;
            } break;
        }

        if (batch->reduction_scratch && plan->stride_bytes == -1) {
            ctx->reg_ptrs[i] = &batch->reduction_scratch[tid * MF_MAX_REGISTERS + i];
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
    
    mf_exec_ctx_init(&state->ctx, (mf_allocator*)&state->temp_arena);
    
    state->ctx.batch_size = (u32)count;
    state->ctx.ndim = batch->ndim; 
    
    if (batch->main_state) {
        state->ctx.global_error_ptr = batch->main_state->global_error_ptr ? 
                                      batch->main_state->global_error_ptr : 
                                      &batch->main_state->error_code;
    }

    state->ctx.linear_offset = (u32)start_idx;
    state->ctx.job_idx = job_idx;
    state->ctx.sync_pass = batch->sync_pass;
    state->ctx.sync_data = batch->sync_data;

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

// --- Sync Ops Detection ---

static bool is_sync_op(u16 opcode) {
    switch(opcode) {
        case MF_OP_CUMSUM:
        case MF_OP_COMPRESS:
            return true;
        default:
            return false;
    }
}

// --- Dispatch ---

static void mf_backend_cpu_dispatch_batch(
    mf_backend_cpu_state* state,
    mf_cpu_parallel_batch* batch,
    uint32_t start_inst,
    uint32_t inst_count
) {
    if (inst_count == 0) return;
    
    batch->start_inst = start_inst;
    batch->inst_count = inst_count;
    
    u32 total_jobs = (u32)((batch->total_elements + MF_CPU_JOB_SIZE - 1) / MF_CPU_JOB_SIZE);

    if (batch->total_elements <= MF_CPU_INLINE_THRESHOLD || total_jobs == 1) {
        mf_backend_cpu_worker_state local_worker;
        _Alignas(16) u8 local_heap[MF_MB(4)]; 
        local_worker.thread_idx = 0;
        local_worker.heap_mem = local_heap;
        local_worker.heap_size = sizeof(local_heap);
        mf_arena_init(&local_worker.temp_arena, local_worker.heap_mem, local_worker.heap_size);
        mf_arena_init(&local_worker.reg_arena, local_worker.reg_arena_mem, sizeof(local_worker.reg_arena_mem));
        
        cpu_worker_job(0, &local_worker, batch);
    } else {
        if (state->pool) {
            mf_thread_pool_run(state->pool, total_jobs, cpu_worker_job, batch);
        } else {
            // Fallback for single-thread without pool (rare)
            // Need a worker_state here, but this path is not used in normal conditions
        }
    }
}

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

    int num_threads = state->pool ? mf_thread_pool_get_thread_count(state->pool) : 1;
    
    mf_cpu_parallel_batch batch = {
        .program = program,
        .main_state = main_state,
        .op_table = state->op_table,
        .total_elements = total_elements,
        .ndim = domain->info.ndim,
        .num_threads = num_threads,
        .reduction_scratch = NULL
    };
    memcpy(batch.domain_shape, domain->info.shape, sizeof(u32) * MF_MAX_DIMS);

    // 0. Pre-calculate Shapes and Allocate Memory
    for (uint32_t i = 0; i < program->meta.tensor_count; ++i) {
        mf_tensor* t = &main_state->registers[i];
        if (!t->buffer && program->builtin_ids[i] == MF_BUILTIN_NONE && program->tensor_data[i] == NULL) {
            mf_exec_ctx_resize_tensor(NULL, t, t->info.shape, t->info.ndim);
        }
    }

    // 1. Build Global Execution Plan (Strides)
    bool reg_processed[MF_MAX_REGISTERS] = {0};
    bool has_reductions = false;
    for (uint32_t i = start_inst; i < start_inst + inst_count; ++i) {
        const mf_instruction* inst = &program->code[i];
        for (int r = 0; r < 5; ++r) {
            uint16_t reg_idx = (r == 0) ? inst->dest_idx : 
                               ((r == 1) ? inst->src1_idx : 
                               ((r == 2) ? inst->src2_idx : 
                               ((r == 3) ? inst->src3_idx : inst->src4_idx)));
            if (reg_idx >= program->meta.tensor_count) continue;
            mf_cpu_reg_plan* plan = &batch.plans[reg_idx];
            mf_tensor* main_t = &main_state->registers[reg_idx];
            i32 stride = inst->strides[r];
            if (!reg_processed[reg_idx] || (plan->stride_bytes == 0 && stride != 0)) {
                plan->info = main_t->info;
                if (stride == -1) { plan->stride_bytes = -1; has_reductions = true; }
                else plan->stride_bytes = (ssize_t)stride * (ssize_t)mf_dtype_size(plan->info.dtype);
                if (program->builtin_ids[reg_idx] != MF_BUILTIN_NONE) {
                    plan->type = MF_SRC_GENERATOR;
                    plan->builtin_id = (mf_builtin_id)program->builtin_ids[reg_idx];
                    plan->builtin_axis = program->builtin_axes[reg_idx];
                } else if (main_t->buffer) {
                    plan->type = MF_SRC_BUFFER; plan->buffer = main_t->buffer; plan->base_offset = main_t->byte_offset;
                } else plan->type = MF_SRC_SCRATCH;
                reg_processed[reg_idx] = true;
            }
        }
    }

    if (has_reductions && num_threads > 1) {
        batch.reduction_scratch = calloc(num_threads * MF_MAX_REGISTERS, sizeof(f32));
    }

    // 2. Linear execution with barrier-splitting
    uint32_t current_batch_start = start_inst;
    for (uint32_t i = start_inst; i < start_inst + inst_count; ++i) {
        if (is_sync_op(program->code[i].opcode)) {
            // Flush preceding tile ops
            if (i > current_batch_start) {
                mf_backend_cpu_dispatch_batch(state, &batch, current_batch_start, i - current_batch_start);
            }
            
            // Execute Sync Op
            if (program->code[i].opcode == MF_OP_CUMSUM) {
                u32 total_jobs = (u32)((batch.total_elements + MF_CPU_JOB_SIZE - 1) / MF_CPU_JOB_SIZE);
                f32* sync_scratch = calloc(total_jobs, sizeof(f32));
                
                // Pass 1: Local Scans
                batch.sync_pass = 0;
                batch.sync_data = sync_scratch;
                mf_backend_cpu_dispatch_batch(state, &batch, i, 1);
                
                // Barrier + Sequential Scan of chunk sums
                f32 global_acc = 0;
                for (u32 j = 0; j < total_jobs; ++j) {
                    f32 chunk_total = sync_scratch[j];
                    sync_scratch[j] = global_acc; // Store OFFSET for this chunk
                    global_acc += chunk_total;
                }
                
                // Pass 2: Global Adjustment
                batch.sync_pass = 1;
                mf_backend_cpu_dispatch_batch(state, &batch, i, 1);
                
                free(sync_scratch);
                batch.sync_pass = 0;
                batch.sync_data = NULL;
            } else {
                // Other sync ops (like Compress) - currently single batch
                mf_backend_cpu_dispatch_batch(state, &batch, i, 1);
            }
            
            current_batch_start = i + 1;
        }
    }

    // Flush last batch
    if (start_inst + inst_count > current_batch_start) {
        mf_backend_cpu_dispatch_batch(state, &batch, current_batch_start, (start_inst + inst_count) - current_batch_start);
    }

    // 3. Merge Reductions
    if (has_reductions && batch.reduction_scratch) {
        for (u32 reg_idx = 0; reg_idx < program->meta.tensor_count; ++reg_idx) {
            if (reg_processed[reg_idx] && batch.plans[reg_idx].stride_bytes == -1) {
                f32 final_val = 0;
                for (int t = 0; t < num_threads; ++t) final_val += batch.reduction_scratch[t * MF_MAX_REGISTERS + reg_idx];
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