#ifndef MF_OPS_ARRAY_H
#define MF_OPS_ARRAY_H

#include <mathflow/isa/mf_dispatch_table.h>

// Registers Array operations (Range, CumSum, etc.) to the dispatch table.
void mf_ops_array_register(mf_backend_dispatch_table* table);

#endif // MF_OPS_ARRAY_H
