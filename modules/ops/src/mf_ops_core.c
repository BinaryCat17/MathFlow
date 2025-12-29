#include <mathflow/ops/mf_ops_core.h>
#include "mf_ops_internal.h"
#include <string.h>

// --- Registration Hub ---

void mf_ops_core_register(mf_op_func* table) {
    mf_ops_register_math(table);
    mf_ops_register_logic(table);
    mf_ops_register_matrix(table);
    mf_ops_register_state(table);
}

void mf_ops_fill_table(mf_op_func* table) {
    if (!table) return;
    memset(table, 0, sizeof(mf_op_func) * MF_OP_LIMIT);
    
    mf_ops_core_register(table);
    mf_ops_array_register(table);
}
