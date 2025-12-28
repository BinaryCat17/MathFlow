#ifndef MF_BACKEND_CPU_H
#define MF_BACKEND_CPU_H

#include <mathflow/isa/mf_dispatch_table.h>

/**
 * @brief Initializes the CPU backend.
 * Creates an internal thread pool and fills the dispatch table.
 * 
 * @param table Pointer to the table to fill.
 * @param num_threads Number of threads (0 = auto).
 */
void mf_backend_cpu_init(mf_backend_dispatch_table* table, int num_threads);

#endif // MF_BACKEND_CPU_H
