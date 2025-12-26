#ifndef MF_BACKEND_CPU_H
#define MF_BACKEND_CPU_H

#include <mathflow/vm/mf_vm.h>

// Populates the dispatch table with CPU implementations
void mf_backend_cpu_init(mf_backend_dispatch_table* table);

#endif // MF_BACKEND_CPU_H
