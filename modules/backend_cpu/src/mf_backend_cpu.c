#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/ops/mf_ops_array.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/base/mf_thread_pool.h>
#include <stdlib.h>
#include <string.h>

// --- Internal Structures ---

typedef struct {
    mf_thread_pool* pool;
} mf_backend_cpu_state;

typedef struct {
    mf_vm vm;
    mf_heap heap;
    void* heap_mem;
    size_t heap_size;
    mf_arena reg_arena;
    u8 reg_arena_mem[4096];
} mf_backend_cpu_worker_state;

typedef struct {
    const mf_context* ctx;
    const mf_vm* main_vm;
    u32 count_x;
    u32 count_y;
} mf_cpu_parallel_batch;

// --- Worker Lifecycle (Internal) ---

static void* worker_init(int thread_idx, void* user_data) {
    (void)thread_idx; (void)user_data;
    
    mf_backend_cpu_worker_state* state = malloc(sizeof(mf_backend_cpu_worker_state));
    
    // Default heap size per thread
    size_t heap_size = 16 * 1024 * 1024;
    state->heap_mem = malloc(heap_size);
    state->heap_size = heap_size;
    mf_heap_init(&state->heap, state->heap_mem, heap_size);
    
    mf_arena_init(&state->reg_arena, state->reg_arena_mem, sizeof(state->reg_arena_mem));
    
    return state;
}

static void worker_cleanup(void* thread_local_data, void* user_data) {
    (void)user_data;
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    free(state->heap_mem);
    free(state);
}

// --- Propagation Logic ---

static void propagate_state(mf_backend_cpu_worker_state* state, const mf_cpu_parallel_batch* batch, u32 job_idx) {
    mf_vm* worker_vm = &state->vm;
    const mf_vm* main_vm = batch->main_vm;
    
    if (!main_vm) return;

    for (size_t i = 0; i < worker_vm->register_count; ++i) {
        if (i >= main_vm->register_count) break;

        mf_tensor* main_t = &main_vm->registers[i];
        mf_tensor* worker_t = &worker_vm->registers[i];

        bool shapes_match = mf_tensor_same_shape(main_t, worker_t);

        if (shapes_match) {
            size_t size = mf_tensor_size_bytes(main_t);
            if (size > 0 && main_t->data) {
                void* data = mf_heap_alloc((mf_allocator*)&state->heap, size);
                if (data) {
                    memcpy(data, main_t->data, size);
                    worker_t->data = data;
                }
            }
            continue;
        }

        // Output Slicing
        bool try_slice_2d = (batch->count_y > 1 && batch->count_x > 1);
        bool try_slice_1d_y = (batch->count_y > 1 && batch->count_x == 1);
        bool try_slice_1d_x = (batch->count_y == 1 && batch->count_x > 1);

        if (try_slice_2d && main_t->ndim == worker_t->ndim + 2) {
            if (main_t->shape[0] == (int32_t)batch->count_y && main_t->shape[1] == (int32_t)batch->count_x) {
                bool suffix_match = true;
                for (int d = 0; d < worker_t->ndim; ++d) if (main_t->shape[d+2] != worker_t->shape[d]) suffix_match = false;
                
                if (suffix_match && main_t->data) {
                    size_t elem_size = mf_tensor_size_bytes(worker_t); 
                    worker_t->data = (u8*)main_t->data + (job_idx * elem_size);
                }
            }
        }
        else if ((try_slice_1d_y || try_slice_1d_x) && main_t->ndim == worker_t->ndim + 1) {
            u32 dim_size = try_slice_1d_y ? batch->count_y : batch->count_x;
            if (main_t->shape[0] == (int32_t)dim_size) {
                bool suffix_match = true;
                for (int d = 0; d < worker_t->ndim; ++d) if (main_t->shape[d+1] != worker_t->shape[d]) suffix_match = false;

                if (suffix_match && main_t->data) {
                    size_t elem_size = mf_tensor_size_bytes(worker_t); 
                    worker_t->data = (u8*)main_t->data + (job_idx * elem_size);
                }
            }
        }
    }
}

static void unbind_external_memory(mf_backend_cpu_worker_state* state) {
    u8* heap_start = (u8*)state->heap_mem;
    u8* heap_end = heap_start + state->heap_size;

    for (size_t i = 0; i < state->vm.register_count; ++i) {
        mf_tensor* t = &state->vm.registers[i];
        if (t->data) {
            u8* ptr = (u8*)t->data;
            if (ptr < heap_start || ptr >= heap_end) {
                t->data = NULL;
            }
        }
    }
}

// --- Job Execution ---

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    mf_cpu_parallel_batch* batch = (mf_cpu_parallel_batch*)user_data;
    
    // 1. Reset VM
    mf_arena_reset(&state->reg_arena);
    mf_vm_init(&state->vm, batch->ctx, (mf_allocator*)&state->heap);
    mf_vm_reset(&state->vm, &state->reg_arena);
    
    // 2. Propagate
    propagate_state(state, batch, job_idx);
    
    // 3. Exec
    mf_vm_exec(&state->vm);
    
    // 4. Cleanup
    unbind_external_memory(state);
    mf_vm_shutdown(&state->vm);
}

// --- Backend API ---

static void mf_backend_cpu_dispatch(
    void* backend_state,
    const struct mf_context* ctx,
    const struct mf_vm* main_vm,
    u32 count_x, u32 count_y
) {
    mf_backend_cpu_state* state = (mf_backend_cpu_state*)backend_state;
    u32 total_jobs = count_x * count_y;
    if (total_jobs == 0) return;

    mf_cpu_parallel_batch batch = {
        .ctx = ctx,
        .main_vm = main_vm,
        .count_x = count_x,
        .count_y = count_y
    };

    if (state && state->pool && total_jobs > 1) {
        mf_thread_pool_run(state->pool, total_jobs, cpu_worker_job, &batch);
    } else {
        // Serial fallback
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
    
    // Create Internal State
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
    
    // Register Operations
    mf_ops_core_register(table);
    mf_ops_array_register(table);
}