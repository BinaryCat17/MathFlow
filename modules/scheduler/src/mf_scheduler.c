#include <mathflow/scheduler/mf_scheduler.h>
#include <mathflow/platform/mf_platform.h>
#include <stdlib.h>
#include <stdio.h>

struct mf_scheduler {
    int num_threads;
};

typedef struct {
    const mf_context* ctx;
    u32 job_count;
    mf_job_setup_func setup_cb;
    mf_job_finish_func finish_cb;
    void* user_data;
    
    mf_atomic_i32* current_job; // Shared atomic counter
} mf_worker_arg;

static void* worker_entry(void* arg) {
    mf_worker_arg* w = (mf_worker_arg*)arg;
    
    // Per-thread VM resources
    // We allocate a heap for this thread.
    // Size? Let's say 16MB per thread for now. 
    // Ideally this should be configurable.
    size_t heap_size = 16 * 1024 * 1024;
    void* heap_mem = malloc(heap_size);
    
    mf_heap heap;
    mf_heap_init(&heap, heap_mem, heap_size);
    
    mf_vm vm;
    mf_vm_init(&vm, w->ctx, (mf_allocator*)&heap);
    
    // Process jobs
    while (1) {
        // atomic_inc returns OLD value? Check my platform impl.
        // My platform: return atomic_fetch_add(var, 1) + 1; -> Returns NEW value.
        // Wait, let's check platform.c
        // atomic_fetch_add returns OLD. I added +1. So it returns NEW value.
        // So 1, 2, 3...
        
        int32_t job_id = mf_atomic_inc(w->current_job) - 1; // 0-based index
        
        if (job_id >= (int32_t)w->job_count) {
            break; // No more jobs
        }
        
        // Reset VM for new run (clears registers but keeps allocator)
        // Wait, mf_vm_reset allocates new registers in ARENA.
        // We don't have a per-thread arena passed in!
        // We need a temporary arena for the VM registers.
        // Let's alloc a small arena on stack or heap.
        
        u8 arena_mem[4096]; // Enough for metadata
        mf_arena local_arena;
        mf_arena_init(&local_arena, arena_mem, 4096);
        
        mf_vm_reset(&vm, &local_arena);
        
        // Setup
        if (w->setup_cb) {
            w->setup_cb(&vm, (u32)job_id, w->user_data);
        }
        
        // Exec
        mf_vm_exec(&vm);
        
        // Finish
        if (w->finish_cb) {
            w->finish_cb(&vm, (u32)job_id, w->user_data);
        }
        
        // Cleanup registers happens on next reset or end of loop (stack arena dies)
        // But dynamic memory in Heap must be freed?
        // mf_vm_shutdown frees Owning tensors.
        mf_vm_shutdown(&vm);
    }
    
    free(heap_mem);
    return NULL;
}

mf_scheduler* mf_scheduler_create(int num_threads) {
    mf_scheduler* s = malloc(sizeof(mf_scheduler));
    if (num_threads <= 0) {
        s->num_threads = mf_cpu_count();
        // Reserve one for main thread? No, workers are separate.
    } else {
        s->num_threads = num_threads;
    }
    return s;
}

void mf_scheduler_destroy(mf_scheduler* sched) {
    if (sched) free(sched);
}

void mf_scheduler_run(
    mf_scheduler* sched,
    const mf_context* ctx,
    u32 job_count,
    mf_job_setup_func setup_cb,
    mf_job_finish_func finish_cb,
    void* user_data
) {
    if (job_count == 0) return;

    mf_atomic_i32 counter = 0;
    mf_atomic_store(&counter, 0);
    
    mf_worker_arg arg;
    arg.ctx = ctx;
    arg.job_count = job_count;
    arg.setup_cb = setup_cb;
    arg.finish_cb = finish_cb;
    arg.user_data = user_data;
    arg.current_job = &counter;
    
    // Limit threads to job count
    int thread_count = sched->num_threads;
    if (thread_count > (int)job_count) thread_count = job_count;
    
    mf_thread_t* threads = malloc(sizeof(mf_thread_t) * thread_count);
    
    for (int i = 0; i < thread_count; ++i) {
        mf_thread_create(&threads[i], worker_entry, &arg);
    }
    
    for (int i = 0; i < thread_count; ++i) {
        mf_thread_join(threads[i]);
    }
    
    free(threads);
}

int mf_scheduler_get_thread_count(mf_scheduler* sched) {
    return sched ? sched->num_threads : 0;
}
