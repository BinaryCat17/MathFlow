#ifndef MF_OPS_INTERNAL_H
#define MF_OPS_INTERNAL_H

#include <mathflow/ops/mf_ops_core.h>

void mf_ops_register_math(mf_op_func* table);
void mf_ops_register_logic(mf_op_func* table);
void mf_ops_register_matrix(mf_op_func* table);
void mf_ops_register_state(mf_op_func* table);

// Aggregate registrars (internal use only)
void mf_ops_core_register(mf_op_func* table);
void mf_ops_array_register(mf_op_func* table);

#endif // MF_OPS_INTERNAL_H
