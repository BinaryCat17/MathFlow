#include <mathflow/ops/mf_ops_core.h>
#include "mf_ops_internal.h"
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>

/**
 * MathFlow Kernel Registration Hub
 * Uses X-Macros to automatically fill the opcode dispatch table.
 */

// No-operation kernel
void op_NOOP(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    (void)ctx; (void)inst;
}

// Forward declarations for all kernels defined in other modules
#define MF_OPCODE(suffix, value) extern void op_##suffix(mf_exec_ctx* ctx, const struct mf_instruction* inst);
MF_OPCODE_LIST
#undef MF_OPCODE

void mf_ops_fill_table(mf_op_func* table) {
    if (!table) return;
    memset(table, 0, sizeof(mf_op_func) * MF_OP_LIMIT);
    
#define MF_OPCODE(suffix, value) table[MF_OP_##suffix] = op_##suffix;
    MF_OPCODE_LIST
#undef MF_OPCODE
}