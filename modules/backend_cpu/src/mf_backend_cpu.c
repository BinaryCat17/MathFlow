#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/ops/mf_ops_array.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_state.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_thread_pool.h>
#include <stdlib.h>
#include <string.h>

// --- Internal Structures ---

#define MF_CPU_TILE_SIZE 64

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
    u8 reg_arena_mem[8192]; // Slightly larger to accommodate register descriptors
} mf_backend_cpu_worker_state;

typedef struct {
    const mf_program* program;
    mf_state* main_state;
    mf_op_func* op_table;
    u32 width;  
    u32 height; 
    u32 tiles_x;
    u32 tiles_y;
} mf_cpu_parallel_batch;

// --- Worker Lifecycle (Internal) ---

static void* worker_init(int thread_idx, void* user_data) {
    (void)thread_idx; (void)user_data;
    
    mf_backend_cpu_worker_state* state = malloc(sizeof(mf_backend_cpu_worker_state));
    
    // Default heap size per thread
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

// --- Interpreter Loop ---

static void impl_error(void* impl, int error_code) {
    mf_exec_ctx* ctx = (mf_exec_ctx*)impl;
    ctx->error = (mf_exec_error)error_code;
}

static void mf_cpu_exec(mf_exec_ctx* ctx, const mf_program* program, mf_op_func* op_table) {
    if (!program || !ctx || !op_table) return;

    // Setup Kernel Context
    mf_kernel_ctx kernel_ctx = {
        .impl = ctx,
        .map_tensor = (mf_tensor* (*)(void*, u16, mf_access_mode))mf_exec_ctx_map_tensor,
        .resize_tensor = (bool (*)(void*, mf_tensor*, const int32_t*, uint8_t))mf_exec_ctx_resize_tensor,
        .error = impl_error,
        .batch_size = ctx->batch_size
    };
    // Copy Intrinsics
    memcpy(kernel_ctx.global_offset, ctx->global_offset, sizeof(ctx->global_offset));
    memcpy(kernel_ctx.local_size, ctx->local_size, sizeof(ctx->local_size));
    memcpy(kernel_ctx.global_size, ctx->global_size, sizeof(ctx->global_size));

    // Execution Loop
    size_t code_count = program->meta.instruction_count;
    mf_instruction* code = program->code;

    for (size_t i = 0; i < code_count; ++i) {
        if (ctx->error != MF_ERROR_NONE) break;

        mf_instruction inst = code[i];
        if (op_table[inst.opcode]) {
            op_table[inst.opcode](&kernel_ctx, inst.dest_idx, inst.src1_idx, inst.src2_idx);
        }
    }
}

// --- Propagation Logic ---

static void prepare_inputs(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch, u32 tile_idx) {
    (void)tile_idx;
    mf_exec_ctx* worker_ctx = &state->ctx;
    mf_state* main_state = batch->main_state;
    
    if (!main_state) return;

    for (size_t i = 0; i < worker_ctx->register_count; ++i) {
        if (i >= main_state->register_count) break;

        mf_tensor* main_t = &main_state->registers[i];
        mf_tensor* worker_t = &worker_ctx->registers[i];

        // 1. Uniforms / Constants
        if (mf_tensor_same_shape(main_t, worker_t)) {
            size_t size = mf_tensor_size_bytes(main_t);
            if (size > 0 && main_t->data) {
                void* data = mf_arena_alloc((mf_allocator*)&state->temp_arena, size);
                if (data) {
                    memcpy(data, main_t->data, size);
                    worker_t->data = data;
                    worker_t->flags |= MF_TENSOR_DYNAMIC; 
                }
            }
            continue;
        }
    }
}

static void commit_outputs(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch, u32 tile_x, u32 tile_y, u32 active_w, u32 active_h) {
    mf_exec_ctx* worker_ctx = &state->ctx;
    mf_state* main_state = batch->main_state;
    
    if (!main_state) return;

    for (size_t i = 0; i < worker_ctx->register_count; ++i) {
        if (i >= main_state->register_count) break;
        
        mf_tensor* worker_t = &worker_ctx->registers[i];
        mf_tensor* main_t = &main_state->registers[i];

        if (!worker_t->data || !main_t->data) continue;
        
        if (main_t->ndim >= 2 && main_t->shape[0] == (int32_t)batch->height && main_t->shape[1] == (int32_t)batch->width) {
            size_t batch_size = active_w * active_h;
            
            int elem_dims = main_t->ndim - 2; 
            size_t elem_size = mf_dtype_size(main_t->dtype);
            for (int d=0; d<elem_dims; ++d) elem_size *= main_t->shape[2+d];
            
            if (worker_t->size < batch_size) continue; 
            
            u8* src_ptr = (u8*)worker_t->data;
            u8* dst_base = (u8*)main_t->data;
            
            size_t main_row_stride = batch->width * elem_size;
            size_t tile_row_size = active_w * elem_size;
            
            for (u32 y = 0; y < active_h; ++y) {
                u32 global_y = tile_y * MF_CPU_TILE_SIZE + y;
                u32 global_x = tile_x * MF_CPU_TILE_SIZE; 
                
                u8* dst_row = dst_base + (global_y * main_row_stride) + (global_x * elem_size);
                u8* src_row = src_ptr + (y * tile_row_size);
                
                memcpy(dst_row, src_row, tile_row_size);
            }
        }
    }
}

// --- Job Execution ---

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    mf_cpu_parallel_batch* batch = (mf_cpu_parallel_batch*)user_data;
    
    // 1. Calculate Tile Bounds
    u32 tile_y = job_idx / batch->tiles_x;
    u32 tile_x = job_idx % batch->tiles_x;
    
    u32 start_x = tile_x * MF_CPU_TILE_SIZE;
    u32 start_y = tile_y * MF_CPU_TILE_SIZE;
    
    u32 active_w = MF_CPU_TILE_SIZE;
    if (start_x + active_w > batch->width) active_w = batch->width - start_x;
    
    u32 active_h = MF_CPU_TILE_SIZE;
    if (start_y + active_h > batch->height) active_h = batch->height - start_y;
    
    if (active_w == 0 || active_h == 0) return;
    
    u32 batch_size = active_w * active_h;

    // 2. Reset Context
    mf_arena_reset(&state->reg_arena);
    mf_arena_reset(&state->temp_arena); 
    
    // Allocate registers locally (pointing to prototypes initially)
    mf_tensor* local_regs = MF_ARENA_PUSH(&state->reg_arena, mf_tensor, batch->program->meta.tensor_count);
    for (u32 i = 0; i < batch->program->meta.tensor_count; ++i) {
        local_regs[i] = batch->program->tensors[i];
        local_regs[i].data = NULL; // Will be set in prepare_inputs or resize
        local_regs[i].flags = MF_TENSOR_DYNAMIC;
    }

    mf_exec_ctx_init(&state->ctx, local_regs, batch->program->meta.tensor_count, (mf_allocator*)&state->temp_arena);
    
    // 3. Setup Virtual Batching
    state->ctx.batch_size = batch_size;
    state->ctx.global_offset[0] = start_y;
    state->ctx.global_offset[1] = start_x;
    state->ctx.global_offset[2] = 0;
    state->ctx.local_size[0] = active_h;
    state->ctx.local_size[1] = active_w;
    state->ctx.local_size[2] = 1;
    state->ctx.global_size[0] = batch->height;
    state->ctx.global_size[1] = batch->width;
    state->ctx.global_size[2] = 1;
    
    // 4. Propagate Inputs
    prepare_inputs(state, batch, job_idx);
    
    // 5. Exec
    mf_cpu_exec(&state->ctx, batch->program, batch->op_table);
    
    // 6. Commit Outputs
    commit_outputs(state, batch, tile_x, tile_y, active_w, active_h);
}

// --- Backend API ---

static void mf_backend_cpu_dispatch(
    void* backend_state,
    const struct mf_program* program,
    struct mf_state* main_state,
    u32 count_x, u32 count_y
) {
    mf_backend_cpu_state* state = (mf_backend_cpu_state*)backend_state;
    
    // Optimization: Script Mode (Single Threaded, In-Place on Stack Context)
    if (count_x == 1 && count_y == 1) {
        if (main_state && program) {
            mf_exec_ctx stack_ctx;
            mf_exec_ctx_init(&stack_ctx, main_state->registers, main_state->register_count, main_state->allocator);
            
            stack_ctx.batch_size = 1;
            stack_ctx.global_offset[0] = 0; stack_ctx.global_offset[1] = 0; stack_ctx.global_offset[2] = 0;
            stack_ctx.global_size[0] = 1; stack_ctx.global_size[1] = 1; stack_ctx.global_size[2] = 1;
            stack_ctx.local_size[0] = 1; stack_ctx.local_size[1] = 1; stack_ctx.local_size[2] = 1;
            
            mf_cpu_exec(&stack_ctx, program, state->op_table);
        }
        return;
    }
    
    u32 tiles_x = (count_x + MF_CPU_TILE_SIZE - 1) / MF_CPU_TILE_SIZE;
    u32 tiles_y = (count_y + MF_CPU_TILE_SIZE - 1) / MF_CPU_TILE_SIZE;
    u32 total_jobs = tiles_x * tiles_y;
    
    if (total_jobs == 0) return;

    mf_cpu_parallel_batch batch = {
        .program = program,
        .main_state = main_state,
        .op_table = state->op_table,
        .width = count_x,
        .height = count_y,
        .tiles_x = tiles_x,
        .tiles_y = tiles_y
    };

    if (state && state->pool && total_jobs > 1) {
        mf_thread_pool_run(state->pool, total_jobs, cpu_worker_job, &batch);
    } else {
        // Serial fallback (still uses a worker state for heap/temp memory isolation)
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
    
    if (state->pool) {
        mf_thread_pool_destroy(state->pool);
    }
    
    free(state);
}

void mf_backend_cpu_init(mf_backend_dispatch_table* table, int num_threads) {
    memset(table, 0, sizeof(mf_backend_dispatch_table));
    
    mf_backend_cpu_state* state = calloc(1, sizeof(mf_backend_cpu_state));
    
    mf_thread_pool_desc pool_desc = {
        .num_threads = num_threads,
        .init_fn = worker_init,
        .cleanup_fn = worker_cleanup,
        .user_data = NULL
    };
    state->pool = mf_thread_pool_create(&pool_desc);
    
    table->state = state;
    table->shutdown = mf_backend_cpu_shutdown;
    table->dispatch = mf_backend_cpu_dispatch;
    
    mf_ops_core_register(table);
    mf_ops_array_register(table);
    
    memcpy(state->op_table, table->op_table, sizeof(state->op_table));
}