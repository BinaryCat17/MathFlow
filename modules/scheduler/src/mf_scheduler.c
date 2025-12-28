#include <mathflow/scheduler/mf_scheduler.h>
#include <mathflow/base/mf_platform.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

struct mf_scheduler {
    int num_threads;
    mf_thread_t* threads;
    
    // Synchronization
    mf_mutex_t mutex;
    mf_cond_t work_cond; // Signal workers to check for work
    mf_cond_t done_cond; // Signal main thread that batch is done
    
    bool running; // True if scheduler is active, false to exit threads
    
    // Current Batch State
    const mf_context* ctx;
    u32 total_jobs;
    mf_atomic_i32 next_job_idx;    // 0..N
    mf_atomic_i32 completed_count; // 0..N
    
    mf_job_setup_func setup_cb;
    mf_job_finish_func finish_cb;
    void* user_data;
};

static void* worker_entry(void* arg) {
    mf_scheduler* sched = (mf_scheduler*)arg;
    
    // --- One-time Thread Initialization ---
    // Persistent Heap per thread
    size_t heap_size = 16 * 1024 * 1024; // 16 MB
    void* heap_mem = malloc(heap_size);
    if (!heap_mem) return NULL; // Should panic ideally

    mf_heap heap;
    mf_heap_init(&heap, heap_mem, heap_size);
    
    mf_vm vm;
    // Note: We don't have the context yet, so we init VM partially or init inside loop.
    // mf_vm_init requires context. We can init it inside the loop or re-init it.
    // Since ctx changes per batch, we must set it per batch.
    
    // Stack Arena for registers (small, fast)
    u8 arena_mem[4096];
    mf_arena local_arena;
    mf_arena_init(&local_arena, arena_mem, sizeof(arena_mem));

    // --- Worker Loop ---
    while (true) {
        // 1. Wait for work
        mf_mutex_lock(&sched->mutex);
        while (sched->running && 
               (mf_atomic_load(&sched->next_job_idx) >= (int32_t)sched->total_jobs)) {
            mf_cond_wait(&sched->work_cond, &sched->mutex);
        }
        
        if (!sched->running) {
            mf_mutex_unlock(&sched->mutex);
            break;
        }
        mf_mutex_unlock(&sched->mutex);
        
        // 2. Process Jobs
        while (true) {
            int32_t job_id = mf_atomic_inc(&sched->next_job_idx) - 1;
            
            if (job_id >= (int32_t)sched->total_jobs) {
                // No more jobs in this batch
                break;
            }
            
            // Init VM for this job
            // We reuse the Heap (it persists), but we reset the Register Arena.
            mf_arena_reset(&local_arena);
            mf_vm_init(&vm, sched->ctx, (mf_allocator*)&heap);
            mf_vm_reset(&vm, &local_arena);
            
            // Setup
            if (sched->setup_cb) {
                sched->setup_cb(&vm, (u32)job_id, sched->user_data);
            }
            
            // Exec
            mf_vm_exec(&vm);
            
            // Finish
            if (sched->finish_cb) {
                sched->finish_cb(&vm, (u32)job_id, sched->user_data);
            }
            
            // Soft Shutdown (frees owning tensors from Heap)
            mf_vm_shutdown(&vm);
            
            // Report Completion
            int32_t finished = mf_atomic_inc(&sched->completed_count);
            if (finished == (int32_t)sched->total_jobs) {
                // Last worker wakes up the main thread
                mf_mutex_lock(&sched->mutex);
                mf_cond_signal(&sched->done_cond);
                mf_mutex_unlock(&sched->mutex);
            }
        }
    }
    
    // --- Cleanup ---
    free(heap_mem);
    return NULL;
}

mf_scheduler* mf_scheduler_create(int num_threads) {
    mf_scheduler* s = malloc(sizeof(mf_scheduler));
    
    if (num_threads <= 0) {
        s->num_threads = mf_cpu_count();
        // Optionally subtract 1 for main thread if we want, but usually full load is fine.
        if (s->num_threads < 1) s->num_threads = 1;
    } else {
        s->num_threads = num_threads;
    }
    
    s->running = true;
    s->threads = malloc(sizeof(mf_thread_t) * s->num_threads);
    
    // Init Sync
    mf_mutex_init(&s->mutex);
    mf_cond_init(&s->work_cond);
    mf_cond_init(&s->done_cond);
    
    // Init State
    s->ctx = NULL;
    s->total_jobs = 0;
    // Set next_job >= total_jobs initially so workers sleep
    mf_atomic_store(&s->next_job_idx, 0); 
    mf_atomic_store(&s->completed_count, 0);
    
    // Launch Threads
    for (int i = 0; i < s->num_threads; ++i) {
        mf_thread_create(&s->threads[i], worker_entry, s);
    }
    
    return s;
}

void mf_scheduler_destroy(mf_scheduler* sched) {
    if (!sched) return;
    
    // Signal Stop
    mf_mutex_lock(&sched->mutex);
    sched->running = false;
    mf_cond_broadcast(&sched->work_cond);
    mf_mutex_unlock(&sched->mutex);
    
    // Join
    for (int i = 0; i < sched->num_threads; ++i) {
        mf_thread_join(sched->threads[i]);
    }
    
    free(sched->threads);
    mf_mutex_destroy(&sched->mutex);
    mf_cond_destroy(&sched->work_cond);
    mf_cond_destroy(&sched->done_cond);
    free(sched);
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
    
    mf_mutex_lock(&sched->mutex);
    
    // Setup Batch
    sched->ctx = ctx;
    sched->setup_cb = setup_cb;
    sched->finish_cb = finish_cb;
    sched->user_data = user_data;
    
    sched->total_jobs = job_count;
    mf_atomic_store(&sched->next_job_idx, 0);
    mf_atomic_store(&sched->completed_count, 0);
    
    // Wake workers
    mf_cond_broadcast(&sched->work_cond);
    
    // Wait for completion
    // We loop to handle spurious wakeups
    while (mf_atomic_load(&sched->completed_count) < (int32_t)job_count) {
        mf_cond_wait(&sched->done_cond, &sched->mutex);
    }
    
    mf_mutex_unlock(&sched->mutex);
}

int mf_scheduler_get_thread_count(mf_scheduler* sched) {
    return sched ? sched->num_threads : 0;
}
