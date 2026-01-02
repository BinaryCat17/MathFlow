#ifndef MF_OPS_CORE_H
#define MF_OPS_CORE_H

#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_instruction.h>
#include <mathflow/base/mf_types.h>

#include <mathflow/isa/mf_exec_ctx.h>

/**
 * @brief Function signature for a MathFlow Operation Kernel (CPU Interpreter).
 */
typedef void (*mf_op_func)(struct mf_exec_ctx* ctx, const struct mf_instruction* inst);

// Registers all available operations to the table.
void mf_ops_fill_table(mf_op_func* table);

#endif // MF_OPS_CORE_H