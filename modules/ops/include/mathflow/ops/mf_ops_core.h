#ifndef MF_OPS_CORE_H
#define MF_OPS_CORE_H

#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_types.h>

struct mf_exec_ctx;

/**
 * @brief Function signature for a MathFlow Operation Kernel (CPU Interpreter).
 */
typedef void (*mf_op_func)(struct mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx);

// Registers all available operations to the table.
void mf_ops_fill_table(mf_op_func* table);

#endif // MF_OPS_CORE_H
