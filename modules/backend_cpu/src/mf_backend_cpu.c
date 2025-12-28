#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/ops/mf_ops_array.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/base/mf_thread_pool.h>
#include <stdlib.h>
#include <string.h>

// --- CPU Worker State ---

typedef struct {
    mf_vm vm;
    mf_heap heap;
    void* heap_mem;
    mf_arena reg_arena;
    u8 reg_arena_mem[4096]; // Fixed size for register metadata
} mf_backend_cpu_worker_state;

typedef struct {
    const mf_context* ctx;
    mf_vm_job_setup_func setup_cb;
    mf_vm_job_finish_func finish_cb;
    void* user_data;
} mf_cpu_parallel_batch;

void* mf_backend_cpu_worker_init(int thread_idx, void* user_data) {
    (void)thread_idx; (void)user_data;
    
    mf_backend_cpu_worker_state* state = malloc(sizeof(mf_backend_cpu_worker_state));
    
    // Default heap size per thread
    size_t heap_size = 16 * 1024 * 1024;
    state->heap_mem = malloc(heap_size);
    mf_heap_init(&state->heap, state->heap_mem, heap_size);
    
    mf_arena_init(&state->reg_arena, state->reg_arena_mem, sizeof(state->reg_arena_mem));
    
    return state;
}

void mf_backend_cpu_worker_cleanup(void* thread_local_data, void* user_data) {
    (void)user_data;
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    free(state->heap_mem);
    free(state);
}

// --- Dispatch Implementation ---

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_backend_cpu_worker_state* state = (mf_backend_cpu_worker_state*)thread_local_data;
    mf_cpu_parallel_batch* batch = (mf_cpu_parallel_batch*)user_data;
    
    // 1. Reset VM for this job
    mf_arena_reset(&state->reg_arena);
    mf_vm_init(&state->vm, batch->ctx, (mf_allocator*)&state->heap);
    mf_vm_reset(&state->vm, &state->reg_arena);
    
    // 2. Setup
    if (batch->setup_cb) {
        batch->setup_cb(&state->vm, job_idx, batch->user_data);
    }
    
    // 3. Execute
    mf_vm_exec(&state->vm);
    
    // 4. Finish
    if (batch->finish_cb) {
        batch->finish_cb(&state->vm, job_idx, batch->user_data);
    }
    
    // 5. Soft Shutdown (frees tensors from Heap, keeping Heap alive)
    mf_vm_shutdown(&state->vm);
}

static void mf_backend_cpu_dispatch(
    const struct mf_context* ctx,
    void* pool, 
    u32 count_x, u32 count_y,
    mf_vm_job_setup_func setup_cb,
    mf_vm_job_finish_func finish_cb,
    void* user_data
) {
    u32 total_jobs = count_x * count_y;
    if (total_jobs == 0) return;

    mf_cpu_parallel_batch batch = {
        .ctx = ctx,
        .setup_cb = setup_cb,
        .finish_cb = finish_cb,
        .user_data = user_data
    };

    if (pool && total_jobs > 1) {
        // Parallel execution
        mf_thread_pool_run((mf_thread_pool*)pool, total_jobs, cpu_worker_job, &batch);
    } else {
        // Serial execution (use a temp VM on stack or just use the caller's thread)
        // Note: For serial, we need a VM. We can't use a thread-local one easily here 
        // without a pool. So we create a small one on stack.
        
        mf_vm vm;
        u8 reg_arena_mem[4096];
        mf_arena reg_arena;
        mf_arena_init(&reg_arena, reg_arena_mem, sizeof(reg_arena_mem));
        
        // We need a heap. For a single job, we can use a temporary heap or just a simple allocator.
        // Let's use a 16MB stack or malloc-ed heap for simplicity.
        size_t heap_size = 16 * 1024 * 1024;
        void* heap_mem = malloc(heap_size);
        mf_heap heap;
        mf_heap_init(&heap, heap_mem, heap_size);

        for (u32 i = 0; i < total_jobs; ++i) {
            mf_arena_reset(&reg_arena);
            mf_vm_init(&vm, ctx, (mf_allocator*)&heap);
            mf_vm_reset(&vm, &reg_arena);
            
            if (setup_cb) setup_cb(&vm, i, user_data);
            mf_vm_exec(&vm);
            if (finish_cb) finish_cb(&vm, i, user_data);
            
            mf_vm_shutdown(&vm);
        }
        
        free(heap_mem);
    }
}

// --- Initialization ---

void mf_backend_cpu_init(mf_backend_dispatch_table* table) {
    memset(table, 0, sizeof(mf_backend_dispatch_table));
    
    // Register Operations
    mf_ops_core_register(table);
    mf_ops_array_register(table);
    
    // Register Dispatch
    table->dispatch = mf_backend_cpu_dispatch;
}
