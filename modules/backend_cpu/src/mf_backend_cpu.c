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
    
    if (current_dim == total_dims - 1) {
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

static void copy_tile_to_global(
    const mf_tensor* tile_t, mf_tensor* global_t, 
    const u32* tile_offset, const u32* active_size
) {
    void* tile_data = mf_tensor_data(tile_t);
    void* global_data = mf_tensor_data(global_t);
    if (!tile_data || !global_data) return;
    
    size_t elem_size = mf_dtype_size(global_t->info.dtype);
    
    // Strides in Bytes
    u32 src_strides[MF_MAX_DIMS];
    u32 dst_strides[MF_MAX_DIMS];
    
    // Src (Tile) is always packed row-major
    u32 current_stride = (u32)elem_size;
    for (int i = (int)global_t->info.ndim - 1; i >= 0; --i) {
        src_strides[i] = current_stride;
        current_stride *= active_size[i]; 
    }

    // Dst (Global) is packed row-major (for now)
    current_stride = (u32)elem_size;
    for (int i = (int)global_t->info.ndim - 1; i >= 0; --i) {
        dst_strides[i] = current_stride;
        current_stride *= (u32)global_t->info.shape[i];
    }
    
    // Calculate Base Offsets
    u8* src_base = (u8*)tile_data;
    u8* dst_base = (u8*)global_data;
    
    for (int i = 0; i < (int)global_t->info.ndim; ++i) {
        dst_base += tile_offset[i] * dst_strides[i];
    }
    
    if (global_t->info.ndim == 1) {
        memcpy(dst_base, src_base, active_size[0] * elem_size);
    } else {
        copy_tile_nd_recursive(0, global_t->info.ndim, src_base, dst_base, active_size, src_strides, dst_strides);
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

        bool same_shape = mf_tensor_same_shape(main_t, worker_t);
        bool scalar_adapt = (worker_t->info.ndim == 0 && main_t->info.ndim == 1 && main_t->info.shape[0] == 1);

        if (same_shape || scalar_adapt) {
             void* worker_data = mf_tensor_data(worker_t);
             if (worker_data) continue; 
             
             void* main_data = mf_tensor_data(main_t);
             if (main_data) {
                 size_t size = mf_tensor_size_bytes(main_t);
                 void* data = mf_arena_alloc((mf_allocator*)&state->temp_arena, size);
                 if (data) {
                     memcpy(data, main_data, size);
                     
                     // We need to attach this new memory to worker_t.
                     // But worker_t is already a view. We need to create a buffer.
                     mf_buffer* buf = mf_arena_alloc((mf_allocator*)&state->temp_arena, sizeof(mf_buffer));
                     mf_buffer_init_view(buf, data, size);
                     worker_t->buffer = buf;
                     worker_t->byte_offset = 0;
                     
                     if (scalar_adapt) {
                         worker_t->info.ndim = main_t->info.ndim;
                         memcpy(worker_t->info.shape, main_t->info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                     }
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

        if (!mf_tensor_is_valid(worker_t) || !mf_tensor_is_valid(main_t)) continue;

        if (main_t->info.ndim != batch->ndim) continue;
        
        bool matches = true;
        for(int d=0; d<batch->ndim; ++d) {
            if ((u32)main_t->info.shape[d] != batch->domain_shape[d]) { matches = false; break; }
        }
        
        if (matches) {
            copy_tile_to_global(worker_t, main_t, worker_ctx->tile_offset, worker_ctx->tile_size);
        }
    }
}

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    mf_cpu_parallel_batch* batch = (mf_cpu_parallel_batch*)user_data;

    u32 tile_coords[MF_MAX_DIMS];
    unflatten_index(job_idx, batch->ndim, batch->tile_strides, tile_coords);

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

    mf_arena_reset(&state->reg_arena);
    mf_arena_reset(&state->temp_arena);
    
    mf_tensor* local_regs = MF_ARENA_PUSH(&state->reg_arena, mf_tensor, batch->program->meta.tensor_count);
    for (u32 i = 0; i < batch->program->meta.tensor_count; ++i) {
        local_regs[i] = batch->program->tensors[i];
    }

    mf_exec_ctx_init(&state->ctx, local_regs, batch->program->meta.tensor_count, (mf_allocator*)&state->temp_arena);
    
    state->ctx.batch_size = batch_size;
    state->ctx.ndim = batch->ndim;
    memcpy(state->ctx.tile_offset, pixel_offset, sizeof(u32) * MF_MAX_DIMS);
    memcpy(state->ctx.tile_size, active_size, sizeof(u32) * MF_MAX_DIMS);
    memcpy(state->ctx.domain_shape, batch->domain_shape, sizeof(u32) * MF_MAX_DIMS);

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

    u8 ndim = domain->info.ndim;
    if (ndim == 0) ndim = 1; 
    if (ndim > MF_MAX_DIMS) ndim = MF_MAX_DIMS;

    u32 domain_shape[MF_MAX_DIMS];
    u32 tiles_per_dim[MF_MAX_DIMS];
    u32 tile_strides[MF_MAX_DIMS];
    
    u32 total_tiles = 1;
    size_t total_elements = 1;

    for (int i = 0; i < ndim; ++i) {
        domain_shape[i] = (i < (int)domain->info.ndim) ? (u32)domain->info.shape[i] : 1;
        if (domain_shape[i] < 1) domain_shape[i] = 1;
        
        tiles_per_dim[i] = (domain_shape[i] + MF_CPU_TILE_SIZE - 1) / MF_CPU_TILE_SIZE;
        total_elements *= domain_shape[i];
    }
    
    u32 current_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        tile_strides[i] = current_stride;
        current_stride *= tiles_per_dim[i];
    }
    total_tiles = current_stride;

    if (total_elements <= MF_CPU_INLINE_THRESHOLD || total_tiles == 1) {
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
