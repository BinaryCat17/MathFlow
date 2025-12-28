#ifndef MF_THREAD_POOL_H
#define MF_THREAD_POOL_H

#include <mathflow/base/mf_types.h>
#include <mathflow/base/mf_platform.h>

typedef struct mf_thread_pool mf_thread_pool;

/**
 * @brief Callback for thread-local initialization.
 * Called once per worker thread when the pool starts.
 * @return Pointer to thread-local data, passed to job_func.
 */
typedef void* (*mf_thread_init_func)(int thread_idx, void* user_data);

/**
 * @brief Callback for thread-local cleanup.
 * Called once per worker thread before the thread exits.
 */
typedef void (*mf_thread_cleanup_func)(void* thread_local_data, void* user_data);

/**
 * @brief The actual job to execute in parallel.
 * @param job_idx Index of the job [0..total_jobs-1].
 * @param thread_local_data Data returned by mf_thread_init_func for this thread.
 * @param user_data Passed to mf_thread_pool_run.
 */
typedef void (*mf_thread_job_func)(u32 job_idx, void* thread_local_data, void* user_data);

typedef struct mf_thread_pool_desc {
    int num_threads;             ///< Number of workers. 0 for auto (CPU count).
    mf_thread_init_func init_fn;    ///< Optional.
    mf_thread_cleanup_func cleanup_fn; ///< Optional.
    void* user_data;             ///< Passed to init/cleanup.
} mf_thread_pool_desc;

/**
 * @brief Creates a persistent thread pool.
 */
mf_thread_pool* mf_thread_pool_create(const mf_thread_pool_desc* desc);

/**
 * @brief Signals all threads to stop and joins them.
 */
void mf_thread_pool_destroy(mf_thread_pool* pool);

/**
 * @brief Runs a batch of jobs in parallel and blocks until all are finished.
 * @param pool The pool instance.
 * @param job_count Total number of jobs.
 * @param job_fn The function to execute.
 * @param user_data Passed to job_fn.
 */
void mf_thread_pool_run(
    mf_thread_pool* pool,
    u32 job_count,
    mf_thread_job_func job_fn,
    void* user_data
);

/**
 * @brief Returns the number of workers in the pool.
 */
int mf_thread_pool_get_thread_count(mf_thread_pool* pool);

#endif // MF_THREAD_POOL_H
