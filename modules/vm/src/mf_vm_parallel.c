#include <mathflow/vm/mf_vm.h>
#include <stdlib.h>

// Internal state for each worker thread
typedef struct {
    mf_vm vm;
    mf_heap heap;
    void* heap_mem;
    mf_arena reg_arena;
    u8 reg_arena_mem[4096]; // Fixed size for register metadata
} mf_vm_worker_state;

// Internal state for the current parallel batch
typedef struct {
    const mf_context* ctx;
    mf_vm_job_setup_func setup_cb;
    mf_vm_job_finish_func finish_cb;
    void* user_data;
} mf_parallel_batch;

// Callback: Thread Initialization
void* mf_vm_worker_init(int thread_idx, void* user_data) {
    (void)thread_idx;
    (void)user_data;
    
    mf_vm_worker_state* state = malloc(sizeof(mf_vm_worker_state));
    
    // TODO: Make heap size configurable? For now 16MB per thread as in scheduler.
    size_t heap_size = 16 * 1024 * 1024;
    state->heap_mem = malloc(heap_size);
    mf_heap_init(&state->heap, state->heap_mem, heap_size);
    
    mf_arena_init(&state->reg_arena, state->reg_arena_mem, sizeof(state->reg_arena_mem));
    
    // Note: ctx will be set during reset in job_func
    return state;
}

// Callback: Thread Cleanup
void mf_vm_worker_cleanup(void* thread_local_data, void* user_data) {
    (void)user_data;
    mf_vm_worker_state* state = (mf_vm_worker_state*)thread_local_data;
    free(state->heap_mem);
    free(state);
}

// Callback: Job Execution
static void vm_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    mf_vm_worker_state* state = (mf_vm_worker_state*)thread_local_data;
    mf_parallel_batch* batch = (mf_parallel_batch*)user_data;
    
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

void mf_vm_exec_parallel(
    const mf_context* ctx,
    mf_thread_pool* pool,
    u32 job_count,
    mf_vm_job_setup_func setup_cb,
    mf_vm_job_finish_func finish_cb,
    void* user_data
) {
    if (!pool || job_count == 0) return;

    // We need a way to pass the batch-specific data to the workers.
    // The thread pool's init/cleanup was set at creation.
    // The job_func gets the pool-wide user_data.
    
    // WAIT: The thread pool was created with some init/cleanup.
    // But who created the thread pool? The Engine.
    // Does the Engine know how to init VM workers?
    // It should!
    
    mf_parallel_batch batch = {
        .ctx = ctx,
        .setup_cb = setup_cb,
        .finish_cb = finish_cb,
        .user_data = user_data
    };

    // Before we can run, we must ensure the pool has the correct worker init/cleanup.
    // Actually, mf_thread_pool_create is where those are set.
    // If the engine creates the pool, it should pass these functions.
    
    mf_thread_pool_run(pool, job_count, vm_worker_job, &batch);
}
