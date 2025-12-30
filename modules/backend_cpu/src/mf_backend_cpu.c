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

#define MF_CPU_TILE_SIZE 64
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
    
    // N-Dimensional Domain
    u8 ndim;
    u32 domain_shape[MF_MAX_DIMS];
    u32 tiles_per_dim[MF_MAX_DIMS];
    u32 tile_strides[MF_MAX_DIMS]; // For unflattening job_idx
} mf_cpu_parallel_batch;

// --- Utils ---

static void unflatten_index(u32 flat_idx, u8 ndim, const u32* strides, u32* out_coords) {
    for (int i = 0; i < ndim; ++i) {
        out_coords[i] = flat_idx / strides[i];
        flat_idx %= strides[i];
    }
}

// Recursive N-Dim Copy
// Src is always packed (Tile). Dst has strides (Global).
static void copy_tile_nd_recursive(
    u8 current_dim, u8 total_dims,
    u8* src_ptr, u8* dst_ptr,
    const u32* size, const u32* src_strides, const u32* dst_strides
) {
    size_t count = size[current_dim];
    
    // Base Case: Innermost dimension (contiguous copy for optimization)
    // Actually, src is packed, so src_strides[last] is elem_size.
    // dst_strides[last] is also likely elem_size (unless there's padding, which we don't support yet).
    if (current_dim == total_dims - 1) {
        // Optimization: Memcpy row
        // Assumption: Element size is baked into strides/ptr arithmetic, 
        // but here strides are in BYTES.
        memcpy(dst_ptr, src_ptr, count * src_strides[current_dim]); 
        return;
    }

    for (u32 i = 0; i < count; ++i) {
        copy_tile_nd_recursive(
            current_dim + 1, total_dims,
            src_ptr + i * src_strides[current_dim],
            dst_ptr + i * dst_strides[current_dim],
            size, src_strides, dst_strides
        );
    }
}

// Helper to calculate byte strides for a packed tensor
static void calc_packed_strides(u8 ndim, const u32* shape, size_t elem_size, u32* out_strides) {
    u32 stride = elem_size;
    for (int i = ndim - 1; i >= 0; --i) {
        out_strides[i] = stride;
        stride *= shape[i];
    }
}

// Helper to calculate byte strides for a global tensor (from int32 strides)
static void calc_global_strides(u8 ndim, const int32_t* mf_strides, size_t elem_size, u32* out_strides) {
    // MathFlow strides are in Elements? No, mf_tensor strides are usually implicitly calculated or passed.
    // mf_tensor struct has int32 strides. Let's assume they are valid if set.
    // But currently we don't set them everywhere. 
    // Standard row-major layout:
    // We need to re-calculate them based on the GLOBAL shape, not the tile shape.
    // Wait, mf_tensor doesn't fully enforce strides yet. Let's assume compact row-major for now.
    // TODO: Use tensor->strides when fully implemented.
    (void)mf_strides;
}

static void copy_tile_to_global(
    const mf_tensor* tile_t, mf_tensor* global_t, 
    const u32* tile_offset, const u32* active_size
) {
    if (!tile_t->data || !global_t->data) return;
    
    size_t elem_size = mf_dtype_size(global_t->dtype);
    
    // Strides in Bytes
    u32 src_strides[MF_MAX_DIMS];
    u32 dst_strides[MF_MAX_DIMS];
    
    // Src (Tile) is always packed row-major
    u32 current_stride = elem_size;
    for (int i = global_t->ndim - 1; i >= 0; --i) {
        src_strides[i] = current_stride;
        // Tile stride depends on the TILE SIZE (e.g. 64), not active size?
        // No, the tile tensor is resized to active_size in this implementation?
        // Actually, let's assume the tile tensor is strictly (active_size).
        current_stride *= active_size[i]; 
    }

    // Dst (Global) is packed row-major (for now)
    current_stride = elem_size;
    for (int i = global_t->ndim - 1; i >= 0; --i) {
        dst_strides[i] = current_stride;
        current_stride *= global_t->shape[i];
    }
    
    // Calculate Base Offsets
    u8* src_base = (u8*)tile_t->data;
    u8* dst_base = (u8*)global_t->data;
    
    for (int i = 0; i < global_t->ndim; ++i) {
        dst_base += tile_offset[i] * dst_strides[i];
    }
    
    // Perform Copy
    // If 1D, simple memcpy
    if (global_t->ndim == 1) {
        memcpy(dst_base, src_base, active_size[0] * elem_size);
    } else {
        // Using "src_strides[ndim-1]" as the element size for the memcpy inside
        // For the recursive function, strides should be step-per-index.
        // Yes, src_strides[last] == elem_size.
        copy_tile_nd_recursive(0, global_t->ndim, src_base, dst_base, active_size, src_strides, dst_strides);
    }
}

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
        mf_instruction inst = code[i];
        if (op_table[inst.opcode]) {
            op_table[inst.opcode](ctx, inst.dest_idx, inst.src1_idx, inst.src2_idx);
        }
    }
}

static void prepare_inputs(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch) {
    mf_exec_ctx* worker_ctx = &state->ctx;
    mf_state* main_state = batch->main_state;
    if (!main_state) return;

    for (size_t i = 0; i < worker_ctx->register_count; ++i) {
        if (i >= main_state->register_count) break;
        mf_tensor* main_t = &main_state->registers[i];
        mf_tensor* worker_t = &worker_ctx->registers[i];

        // Propagate Uniforms (Scalar or matching shape)
        // Allow Scalar (0-rank) <-> Vector[1] (1-rank) adaptation
        bool same_shape = mf_tensor_same_shape(main_t, worker_t);
        bool scalar_adapt = (worker_t->ndim == 0 && main_t->ndim == 1 && main_t->shape[0] == 1);

        if (same_shape || scalar_adapt) {
             if (worker_t->data) continue; 
             // Copy data from main state
             if (main_t->data) {
                 size_t size = mf_tensor_size_bytes(main_t);
                 void* data = mf_arena_alloc((mf_allocator*)&state->temp_arena, size);
                 if (data) {
                     memcpy(data, main_t->data, size);
                     worker_t->data = data;
                     
                     // Adopt shape from main state if adapting
                     if (scalar_adapt) {
                         worker_t->ndim = main_t->ndim;
                         memcpy(worker_t->shape, main_t->shape, sizeof(int32_t) * MF_MAX_DIMS);
                         worker_t->size = main_t->size;
                     }

                     worker_t->flags |= MF_TENSOR_DYNAMIC;
                 }
             }
        }
    }
}

static void commit_outputs(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch) {
    mf_exec_ctx* worker_ctx = &state->ctx;
    mf_state* main_state = batch->main_state;
    if (!main_state) return;

    for (size_t i = 0; i < worker_ctx->register_count; ++i) {
        if (i >= main_state->register_count) break;
        mf_tensor* worker_t = &worker_ctx->registers[i];
        mf_tensor* main_t = &main_state->registers[i];

        if (!worker_t->data || !main_t->data) continue;

        // Check if this register maps to the Domain (Output)
        // Heuristic: If global tensor shape matches domain shape (and is large), we commit.
        // Or if it matches the kernel bindings.
        // For now, check if global shape >= tile offset
        
        // Only commit if dimensions match (rank)
        if (main_t->ndim != batch->ndim) continue;
        
        // Only commit if it looks like a spatial field (matches domain size)
        bool matches = true;
        for(int d=0; d<batch->ndim; ++d) {
            if (main_t->shape[d] != batch->domain_shape[d]) { matches = false; break; }
        }
        
        if (matches) {
            copy_tile_to_global(worker_t, main_t, worker_ctx->tile_offset, worker_ctx->tile_size);
        }
    }
}

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    mf_cpu_parallel_batch* batch = (mf_cpu_parallel_batch*)user_data;

    // 1. Unflatten Job ID -> Tile Coords
    u32 tile_coords[MF_MAX_DIMS];
    unflatten_index(job_idx, batch->ndim, batch->tile_strides, tile_coords);

    // 2. Calculate Active Region
    u32 pixel_offset[MF_MAX_DIMS];
    u32 active_size[MF_MAX_DIMS];
    u32 batch_size = 1;

    for (int i = 0; i < batch->ndim; ++i) {
        pixel_offset[i] = tile_coords[i] * MF_CPU_TILE_SIZE;
        u32 size = MF_CPU_TILE_SIZE;
        if (pixel_offset[i] + size > batch->domain_shape[i]) {
            size = batch->domain_shape[i] - pixel_offset[i];
        }
        active_size[i] = size;
        batch_size *= size;
    }

    if (batch_size == 0) return;

    // 3. Reset Context
    mf_arena_reset(&state->reg_arena);
    mf_arena_reset(&state->temp_arena);
    
    // 4. Setup Local Registers
    mf_tensor* local_regs = MF_ARENA_PUSH(&state->reg_arena, mf_tensor, batch->program->meta.tensor_count);
    for (u32 i = 0; i < batch->program->meta.tensor_count; ++i) {
        local_regs[i] = batch->program->tensors[i];
        if (local_regs[i].data == NULL) {
             local_regs[i].flags = MF_TENSOR_DYNAMIC;
        } else {
             local_regs[i].flags &= ~MF_TENSOR_DYNAMIC;
             local_regs[i].flags &= ~MF_TENSOR_OWNS_DATA;
        }
    }

    mf_exec_ctx_init(&state->ctx, local_regs, batch->program->meta.tensor_count, (mf_allocator*)&state->temp_arena);
    
    // 5. Fill Context
    state->ctx.batch_size = batch_size;
    state->ctx.ndim = batch->ndim;
    memcpy(state->ctx.tile_offset, pixel_offset, sizeof(u32) * MF_MAX_DIMS);
    memcpy(state->ctx.tile_size, active_size, sizeof(u32) * MF_MAX_DIMS);
    memcpy(state->ctx.domain_shape, batch->domain_shape, sizeof(u32) * MF_MAX_DIMS);

    // 6. Propagate & Execute
    prepare_inputs(state, batch);
    mf_cpu_exec(&state->ctx, batch->program, batch->op_table);
    
    if (state->ctx.error != MF_ERROR_NONE && batch->main_state) {
        batch->main_state->error_code = (int32_t)state->ctx.error;
    } else {
        commit_outputs(state, batch);
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

    // 1. Analyze Domain
    u8 ndim = domain->ndim;
    if (ndim == 0) ndim = 1; // Scalar
    if (ndim > MF_MAX_DIMS) ndim = MF_MAX_DIMS;

    u32 domain_shape[MF_MAX_DIMS];
    u32 tiles_per_dim[MF_MAX_DIMS];
    u32 tile_strides[MF_MAX_DIMS];
    
    u32 total_tiles = 1;
    size_t total_elements = 1;

    for (int i = 0; i < ndim; ++i) {
        domain_shape[i] = (i < domain->ndim) ? domain->shape[i] : 1;
        if (domain_shape[i] < 1) domain_shape[i] = 1;
        
        tiles_per_dim[i] = (domain_shape[i] + MF_CPU_TILE_SIZE - 1) / MF_CPU_TILE_SIZE;
        total_elements *= domain_shape[i];
    }
    
    // Calculate Tile Strides (Row-Major: Last dim is 1)
    // Actually, job indices map to TILE COORDINATES.
    // e.g. TILE_COORDS [0,0] -> Job 0.
    // Strides for unflattening:
    // If dims=[Y, X], strides=[TilesX, 1].
    // Coords[0] = idx / TilesX.
    
    u32 current_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        tile_strides[i] = current_stride;
        current_stride *= tiles_per_dim[i];
    }
    total_tiles = current_stride;

    // 2. Fast Path (Inline)
    if (total_elements <= MF_CPU_INLINE_THRESHOLD || total_tiles == 1) {
        // Run on calling thread using temporary worker state
        // TODO: Reuse a cached thread-local state? For now, alloc/free is okay for "Fast Path" 
        // compared to ThreadPool overhead, but ideally we want zero alloc.
        // Actually, creating a worker state (16MB heap) is expensive!
        // We should use a simplified stack-based execution for extremely small jobs.
        
        // For Phase 20 MVP, let's just use the standard path but serial.
        // Optimization: if truly scalar (1 element), stack allocate registers?
        
        mf_backend_cpu_worker_state* temp_worker = worker_init(0, NULL);
        
        mf_cpu_parallel_batch batch = {
            .program = program,
            .main_state = main_state,
            .op_table = state->op_table,
            .ndim = ndim,
        };
        memcpy(batch.domain_shape, domain_shape, sizeof(domain_shape));
        memcpy(batch.tiles_per_dim, tiles_per_dim, sizeof(tiles_per_dim));
        memcpy(batch.tile_strides, tile_strides, sizeof(tile_strides));
        
        cpu_worker_job(0, temp_worker, &batch);
        
        worker_cleanup(temp_worker, NULL);
        return;
    }

    // 3. Parallel Dispatch
    mf_cpu_parallel_batch batch = {
        .program = program,
        .main_state = main_state,
        .op_table = state->op_table,
        .ndim = ndim,
    };
    memcpy(batch.domain_shape, domain_shape, sizeof(domain_shape));
    memcpy(batch.tiles_per_dim, tiles_per_dim, sizeof(tiles_per_dim));
    memcpy(batch.tile_strides, tile_strides, sizeof(tile_strides));

    if (state->pool) {
        mf_thread_pool_run(state->pool, total_tiles, cpu_worker_job, &batch);
    } else {
        mf_backend_cpu_worker_state* temp_worker = worker_init(0, NULL);
        for (u32 i = 0; i < total_tiles; ++i) {
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