#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include "mf_ops_internal.h"
#include <string.h>

/**
 * MathFlow Logic Kernels
 * Automatically generated from mf_ops_db.inc
 */

#define MF_GEN_AUTO(_op, _ke, _ar) MF_KERNEL_AUTO(_op, _ke, _ar)
#define MF_GEN_MANUAL(...)

#define MF_OP(_s, _n, _op, _cat, _in, _out, _tr, _sr, _ar, _p1, _p2, _p3, _p4, _kt, _ke, _arity) \
    MF_GEN_##_kt(_op, _ke, _arity)

MF_OP_LIST

#undef MF_OP
#undef MF_GEN_AUTO
#undef MF_GEN_MANUAL
