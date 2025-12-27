#ifndef MF_OPS_CORE_H
#define MF_OPS_CORE_H

#include <mathflow/vm/mf_vm.h>

// Registers Core Math operations (Add, Sub, Sin, Cos, etc.) to the dispatch table.
void mf_ops_core_register(mf_backend_dispatch_table* table);

#endif // MF_OPS_CORE_H
