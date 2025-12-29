#ifndef MF_OPS_INTERNAL_H
#define MF_OPS_INTERNAL_H

#include <mathflow/isa/mf_dispatch_table.h>

void mf_ops_register_math(mf_backend_dispatch_table* table);
void mf_ops_register_logic(mf_backend_dispatch_table* table);
void mf_ops_register_matrix(mf_backend_dispatch_table* table);
void mf_ops_register_state(mf_backend_dispatch_table* table);

#endif // MF_OPS_INTERNAL_H
