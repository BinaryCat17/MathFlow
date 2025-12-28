#ifndef MF_BACKEND_CPU_H
#define MF_BACKEND_CPU_H

#include <mathflow/isa/mf_dispatch_table.h>

/**
 * @brief Initializes the CPU backend.
 * Fills the dispatch table with CPU-specific kernel pointers and the dispatch function.
 */
void mf_backend_cpu_init(mf_backend_dispatch_table* table);

/**
 * @brief Worker initialization for CPU threads using VMs.
 * Internal but exposed for mf_engine to setup its thread pool.
 */
void* mf_backend_cpu_worker_init(int thread_idx, void* user_data);

/**
 * @brief Worker cleanup for CPU threads.
 */
void mf_backend_cpu_worker_cleanup(void* thread_local_data, void* user_data);

#endif // MF_BACKEND_CPU_H
