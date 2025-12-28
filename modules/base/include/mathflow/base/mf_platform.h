#ifndef MF_PLATFORM_H
#define MF_PLATFORM_H

#include <stdint.h>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    
    typedef HANDLE mf_thread_t;
    typedef CRITICAL_SECTION mf_mutex_t;
    typedef CONDITION_VARIABLE mf_cond_t;
    
    // Simple atomic for counter
    typedef volatile LONG mf_atomic_i32;

#else
    #include <pthread.h>
    #include <unistd.h>
    #include <stdatomic.h>
    
    typedef pthread_t mf_thread_t;
    typedef pthread_mutex_t mf_mutex_t;
    typedef pthread_cond_t mf_cond_t;
    
    typedef atomic_int mf_atomic_i32;
#endif

// Thread Function Prototype
typedef void* (*mf_thread_func)(void* arg);

// --- Thread API ---
int mf_thread_create(mf_thread_t* thread, mf_thread_func func, void* arg);
int mf_thread_join(mf_thread_t thread);
int mf_cpu_count(void);

// --- Mutex API ---
void mf_mutex_init(mf_mutex_t* mutex);
void mf_mutex_lock(mf_mutex_t* mutex);
void mf_mutex_unlock(mf_mutex_t* mutex);
void mf_mutex_destroy(mf_mutex_t* mutex);

// --- CondVar API ---
void mf_cond_init(mf_cond_t* cond);
void mf_cond_wait(mf_cond_t* cond, mf_mutex_t* mutex);
void mf_cond_signal(mf_cond_t* cond);
void mf_cond_broadcast(mf_cond_t* cond);
void mf_cond_destroy(mf_cond_t* cond);

// --- Atomic API ---
int32_t mf_atomic_inc(mf_atomic_i32* var);
int32_t mf_atomic_load(mf_atomic_i32* var);
void mf_atomic_store(mf_atomic_i32* var, int32_t val);

#endif // MF_PLATFORM_H
