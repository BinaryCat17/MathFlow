#include <mathflow/ops/mf_ops_core.h>
#include "mf_ops_internal.h"

// --- Registration Hub ---

void mf_ops_core_register(mf_backend_dispatch_table* table) {
    mf_ops_register_math(table);
    mf_ops_register_logic(table);
    mf_ops_register_matrix(table);
    mf_ops_register_state(table);
}
