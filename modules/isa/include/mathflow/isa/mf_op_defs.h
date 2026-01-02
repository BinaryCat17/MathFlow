#ifndef MF_OP_DEFS_H
#define MF_OP_DEFS_H

#include <mathflow/base/mf_types.h>

/**
 * MathFlow Operation Definitions (Metadata Structures)
 */

typedef enum {
    MF_OP_CAT_SPECIAL,     // Compiler intrinsic (Const, Input, Call, Copy, Output)
    MF_OP_CAT_ATOMIC,      // Primitive Math/Logic (1:1 element mapping)
    MF_OP_CAT_REDUCTION,   // Data reduction (Sum, Size, CumSum)
    MF_OP_CAT_ACCEL,       // High-performance accelerators (MatMul, Inverse)
    MF_OP_CAT_MEMORY,      // Layout & Random Access (Gather, Slice, Reshape, Filter)
} mf_op_category;

#define MF_TYPE_MASK_F32 (1 << MF_DTYPE_F32)
#define MF_TYPE_MASK_I32 (1 << MF_DTYPE_I32)
#define MF_TYPE_MASK_U8  (1 << MF_DTYPE_U8)
#define MF_TYPE_MASK_NUMERIC (MF_TYPE_MASK_F32 | MF_TYPE_MASK_I32)
#define MF_TYPE_MASK_ALL     (MF_TYPE_MASK_NUMERIC | MF_TYPE_MASK_U8)
#define MF_TYPE_MASK_LOGIC   (MF_TYPE_MASK_U8)

typedef enum {
    MF_OUT_SAME_AS_INPUT,   // Output follows s1 dtype (default)
    MF_OUT_SAME_AS_INPUT_2, // Output follows s2 dtype
    MF_OUT_FORCE_F32,       // Always F32
    MF_OUT_FORCE_U8,        // Always U8
    MF_OUT_FORCE_I32,       // Always I32
} mf_out_rule;

typedef enum {
    MF_SHAPE_SPECIAL,       // Handled individually (Const, Input, Call)
    MF_SHAPE_SAME_AS_S1,    // Output shape = Input 1 shape
    MF_SHAPE_SAME_AS_S2,    // Output shape = Input 2 shape
    MF_SHAPE_BROADCAST,     // Broadcast S1 and S2 (and S3 if present)
    MF_SHAPE_MATMUL,        // Matrix Multiplication [M,K] x [K,N] -> [M,N]
    MF_SHAPE_TRANSPOSE,     // Swap dim 0 and 1
    MF_SHAPE_DOT,           // Dot product (reduces last dim)
    MF_SHAPE_JOIN,          // Join/Concat (adds dimension)
    MF_SHAPE_GATHER,        // Shape follows indices
    MF_SHAPE_RESHAPE,       // Shape follows constant value
    MF_SHAPE_SLICE,         // 1D slice
    MF_SHAPE_SCALAR,        // Output is a single value (ndim=0)
} mf_shape_rule;

typedef enum {
    MF_ACCESS_LINEAR,       // 1:1 element-wise mapping
    MF_ACCESS_WINDOW,       // Neighborhood access (Stencil/Relative)
    MF_ACCESS_RANDOM,       // Indirect access (Gather/Scatter)
    MF_ACCESS_GLOBAL,       // Full buffer access (Reductions)
    MF_ACCESS_SPECIAL,      // Handled by compiler (Const, Input, Call)
} mf_access_pattern;

#include "mf_ops_db.inc"

#endif // MF_OP_DEFS_H