#ifndef MF_SCHEDULER_H
#define MF_SCHEDULER_H

#include <mathflow/vm/mf_vm.h>
#include <mathflow/base/mf_platform.h>

typedef struct mf_scheduler mf_scheduler;

// Callback setup function called before execution on a worker thread.
// Use this to set input tensors for a specific tile/job.
// - vm: The thread-local VM instance (freshly reset).
// - job_idx: Index of the job [0..job_count-1].
// - user_data: Passed from run().
typedef void (*mf_job_setup_func)(mf_vm* vm, u32 job_idx, void* user_data);

// Callback called after execution (e.g. to copy results).
// - vm: The thread-local VM with results.
typedef void (*mf_job_finish_func)(mf_vm* vm, u32 job_idx, void* user_data);

// --- Scheduler API ---

// Create a scheduler with a specific number of worker threads.
// If num_threads <= 0, uses mf_cpu_count().
mf_scheduler* mf_scheduler_create(int num_threads);

// Destroy scheduler and stop threads.
void mf_scheduler_destroy(mf_scheduler* sched);

// Run a parallel task.
// Blocks until all jobs are completed.
// - ctx: The Read-Only context (Program).
// - job_count: Total number of tasks to split work into.
void mf_scheduler_run(
    mf_scheduler* sched,
    const mf_context* ctx,
    u32 job_count,
    mf_job_setup_func setup_cb,
    mf_job_finish_func finish_cb,
    void* user_data
);

// Get the number of active workers.
int mf_scheduler_get_thread_count(mf_scheduler* sched);

#endif // MF_SCHEDULER_H
