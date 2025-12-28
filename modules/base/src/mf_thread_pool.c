#include <mathflow/base/mf_thread_pool.h>
#include <stdlib.h>
#include <stdbool.h>

struct mf_thread_pool {
    int num_threads;
    mf_thread_t* threads;
    
    // Synchronization
    mf_mutex_t mutex;
    mf_cond_t work_cond; 
    mf_cond_t done_cond;
    
    bool running;
    
    // Batch State
    u32 total_jobs;
    mf_atomic_i32 next_job_idx;
    mf_atomic_i32 completed_count;
    
    mf_thread_job_func job_fn;
    void* job_user_data;

    // Callbacks
    mf_thread_init_func init_fn;
    mf_thread_cleanup_func cleanup_fn;
    void* init_user_data;
};

typedef struct {
    mf_thread_pool* pool;
    int thread_idx;
} worker_arg;

static void* worker_entry(void* arg) {
    worker_arg* warg = (worker_arg*)arg;
    mf_thread_pool* pool = warg->pool;
    int thread_idx = warg->thread_idx;
    free(warg);

    void* thread_local_data = NULL;
    if (pool->init_fn) {
        thread_local_data = pool->init_fn(thread_idx, pool->init_user_data);
    }

    while (true) {
        mf_mutex_lock(&pool->mutex);
        while (pool->running && 
               (mf_atomic_load(&pool->next_job_idx) >= (int32_t)pool->total_jobs)) {
            mf_cond_wait(&pool->work_cond, &pool->mutex);
        }
        
        if (!pool->running) {
            mf_mutex_unlock(&pool->mutex);
            break;
        }
        mf_mutex_unlock(&pool->mutex);
        
        while (true) {
            int32_t job_id = mf_atomic_inc(&pool->next_job_idx) - 1;
            
            if (job_id >= (int32_t)pool->total_jobs) {
                break;
            }
            
            pool->job_fn((u32)job_id, thread_local_data, pool->job_user_data);
            
            int32_t finished = mf_atomic_inc(&pool->completed_count);
            if (finished == (int32_t)pool->total_jobs) {
                mf_mutex_lock(&pool->mutex);
                mf_cond_signal(&pool->done_cond);
                mf_mutex_unlock(&pool->mutex);
            }
        }
    }
    
    if (pool->cleanup_fn) {
        pool->cleanup_fn(thread_local_data, pool->init_user_data);
    }
    
    return NULL;
}

mf_thread_pool* mf_thread_pool_create(const mf_thread_pool_desc* desc) {
    mf_thread_pool* p = malloc(sizeof(mf_thread_pool));
    
    int n = desc->num_threads;
    if (n <= 0) {
        n = mf_cpu_count();
        if (n < 1) n = 1;
    }
    
    p->num_threads = n;
    p->running = true;
    p->threads = malloc(sizeof(mf_thread_t) * n);
    
    mf_mutex_init(&p->mutex);
    mf_cond_init(&p->work_cond);
    mf_cond_init(&p->done_cond);
    
    p->total_jobs = 0;
    mf_atomic_store(&p->next_job_idx, 0); 
    mf_atomic_store(&p->completed_count, 0);
    
    p->init_fn = desc->init_fn;
    p->cleanup_fn = desc->cleanup_fn;
    p->init_user_data = desc->user_data;
    p->job_fn = NULL;
    p->job_user_data = NULL;
    
    for (int i = 0; i < n; ++i) {
        worker_arg* warg = malloc(sizeof(worker_arg));
        warg->pool = p;
        warg->thread_idx = i;
        mf_thread_create(&p->threads[i], worker_entry, warg);
    }
    
    return p;
}

void mf_thread_pool_destroy(mf_thread_pool* pool) {
    if (!pool) return;
    
    mf_mutex_lock(&pool->mutex);
    pool->running = false;
    mf_cond_broadcast(&pool->work_cond);
    mf_mutex_unlock(&pool->mutex);
    
    for (int i = 0; i < pool->num_threads; ++i) {
        mf_thread_join(pool->threads[i]);
    }
    
    free(pool->threads);
    mf_mutex_destroy(&pool->mutex);
    mf_cond_destroy(&pool->work_cond);
    mf_cond_destroy(&pool->done_cond);
    free(pool);
}

void mf_thread_pool_run(
    mf_thread_pool* pool,
    u32 job_count,
    mf_thread_job_func job_fn,
    void* user_data
) {
    if (job_count == 0) return;
    
    mf_mutex_lock(&pool->mutex);
    
    pool->job_fn = job_fn;
    pool->job_user_data = user_data;
    pool->total_jobs = job_count;
    mf_atomic_store(&pool->next_job_idx, 0);
    mf_atomic_store(&pool->completed_count, 0);
    
    mf_cond_broadcast(&pool->work_cond);
    
    while (mf_atomic_load(&pool->completed_count) < (int32_t)job_count) {
        mf_cond_wait(&pool->done_cond, &pool->mutex);
    }
    
    mf_mutex_unlock(&pool->mutex);
}

int mf_thread_pool_get_thread_count(mf_thread_pool* pool) {
    return pool ? pool->num_threads : 0;
}
