#ifndef MF_OP_DEFS_H
#define MF_OP_DEFS_H

/**
 * MathFlow Operation Definitions (Single Source of Truth)
 * 
 * Format:
 * MF_OP(node_suffix, json_name, opcode, category, type_mask, type_rule, shape_rule, access_rule, p1, p2, p3, p4)
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

#define MF_OP_LIST \
    /* --- Special Nodes (Compiler Intrinsics) --- */ \
    MF_OP(CONST,   "Const",   MF_OP_NOOP,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SPECIAL,    MF_ACCESS_SPECIAL, "out", NULL,  NULL,  NULL) \
    MF_OP(INPUT,   "Input",   MF_OP_NOOP,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SPECIAL,    MF_ACCESS_SPECIAL, "out", NULL,  NULL,  NULL) \
    MF_OP(OUTPUT,  "Output",  MF_OP_COPY,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SPECIAL,    MF_ACCESS_LINEAR,  "in",  NULL,  NULL,  NULL) \
    MF_OP(CALL,    "Call",    MF_OP_NOOP,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SPECIAL,    MF_ACCESS_SPECIAL, NULL,  NULL,  NULL,  NULL) \
    MF_OP(COPY,    "Copy",    MF_OP_COPY,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "in",  NULL,  NULL,  NULL) \
    \
    /* --- Atomic Math (1:1 element mapping) --- */ \
    MF_OP(ADD,     "Add",     MF_OP_ADD,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(SUB,     "Sub",     MF_OP_SUB,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(MUL,     "Mul",     MF_OP_MUL,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(DIV,     "Div",     MF_OP_DIV,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(ABS,     "Abs",     MF_OP_ABS,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "x",   NULL,  NULL,  NULL) \
    MF_OP(SIN,     "Sin",     MF_OP_SIN,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "x",   NULL,  NULL,  NULL) \
    MF_OP(COS,     "Cos",     MF_OP_COS,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "x",   NULL,  NULL,  NULL) \
    MF_OP(SQRT,    "Sqrt",    MF_OP_SQRT,    MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "x",   NULL,  NULL,  NULL) \
    MF_OP(FLOOR,   "Floor",   MF_OP_FLOOR,   MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "x",   NULL,  NULL,  NULL) \
    MF_OP(CEIL,    "Ceil",    MF_OP_CEIL,    MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "x",   NULL,  NULL,  NULL) \
    MF_OP(POW,     "Pow",     MF_OP_POW,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "base","exp", NULL,  NULL) \
    MF_OP(ATAN2,   "Atan2",   MF_OP_ATAN2,   MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "y",   "x",   NULL,  NULL) \
    MF_OP(MIN,     "Min",     MF_OP_MIN,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(MAX,     "Max",     MF_OP_MAX,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(FMA,     "Fma",     MF_OP_FMA,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   "c",   NULL) \
    MF_OP(CLAMP,   "Clamp",   MF_OP_CLAMP,   MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "x",   "min", "max", NULL) \
    MF_OP(STEP,    "Step",    MF_OP_STEP,    MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "edge","x",   NULL,  NULL) \
    MF_OP(MIX,     "Mix",     MF_OP_MIX,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   "t",   NULL) \
    MF_OP(SMOOTHSTEP,"SmoothStep",MF_OP_SMOOTHSTEP,MF_OP_CAT_ATOMIC,MF_TYPE_MASK_F32, MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S2, MF_ACCESS_LINEAR,  "edges","x",  NULL,  NULL) \
    MF_OP(SELECT,  "Select",  MF_OP_SELECT,  MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT_2, MF_SHAPE_BROADCAST,MF_ACCESS_LINEAR,  "cond","true","false",NULL) \
    \
    /* --- Atomic Logic --- */ \
    MF_OP(LESS,    "Less",    MF_OP_LESS,    MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(GREATER, "Greater", MF_OP_GREATER, MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(EQUAL,   "Equal",   MF_OP_EQUAL,   MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(NEQUAL,  "NotEqual",MF_OP_NEQUAL,  MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(LEQUAL,  "LessEqual",MF_OP_LEQUAL, MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(GEQUAL,  "GreaterEqual",MF_OP_GEQUAL, MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(AND,     "And",     MF_OP_AND,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_LOGIC,   MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(OR,      "Or",      MF_OP_OR,      MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_LOGIC,   MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(XOR,     "Xor",     MF_OP_XOR,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_LOGIC,   MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  MF_ACCESS_LINEAR,  "a",   "b",   NULL,  NULL) \
    MF_OP(NOT,     "Not",     MF_OP_NOT,     MF_OP_CAT_ATOMIC,  MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_SAME_AS_S1, MF_ACCESS_LINEAR,  "in",  NULL,  NULL,  NULL) \
    \
    /* --- Reductions (N:1 mapping) --- */ \
    MF_OP(REDUCE_SUM, "ReduceSum", MF_OP_SUM, MF_OP_CAT_REDUCTION, MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_SCALAR,    MF_ACCESS_GLOBAL,  "in",  NULL,  NULL,  NULL) \
    MF_OP(DOT,        "Dot",       MF_OP_DOT,     MF_OP_CAT_REDUCTION, MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_DOT,       MF_ACCESS_WINDOW,  "a",   "b",   NULL,  NULL) \
    MF_OP(LENGTH,     "Length",    MF_OP_LENGTH,  MF_OP_CAT_REDUCTION, MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_DOT,       MF_ACCESS_WINDOW,  "x",   NULL,  NULL,  NULL) \
    MF_OP(SIZE,       "Size",      MF_OP_NOOP,    MF_OP_CAT_REDUCTION, MF_TYPE_MASK_ALL,     MF_OUT_FORCE_F32,     MF_SHAPE_SCALAR,    MF_ACCESS_GLOBAL,  "in",  NULL,  NULL,  NULL) \
    MF_OP(CUMSUM,  "CumSum",  MF_OP_CUMSUM,  MF_OP_CAT_REDUCTION, MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, MF_ACCESS_GLOBAL, "in",  NULL,  NULL,  NULL) \
    \
    /* --- Accelerators (Heavy Math) --- */ \
    MF_OP(MATMUL,  "MatMul",    MF_OP_MATMUL,    MF_OP_CAT_ACCEL, MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_MATMUL,    MF_ACCESS_WINDOW,  "a",   "b",   NULL,  NULL) \
    MF_OP(INVERSE, "Inverse",   MF_OP_INVERSE,   MF_OP_CAT_ACCEL, MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, MF_ACCESS_GLOBAL,  "in",  NULL,  NULL,  NULL) \
    \
    /* --- Memory & Layout --- */ \
    MF_OP(TRANSPOSE,"Transpose", MF_OP_TRANSPOSE, MF_OP_CAT_MEMORY, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_TRANSPOSE, MF_ACCESS_LINEAR,  "in",  NULL,  NULL,  NULL) \
    MF_OP(NORMALIZE,"Normalize", MF_OP_NORMALIZE, MF_OP_CAT_MEMORY, MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, MF_ACCESS_WINDOW,  "in",  NULL,  NULL,  NULL) \
    MF_OP(JOIN,    "Join",      MF_OP_JOIN,      MF_OP_CAT_MEMORY, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_JOIN,      MF_ACCESS_LINEAR,  "a",   "b",   "c",   "d") \
    MF_OP(GATHER,  "Gather",  MF_OP_GATHER,  MF_OP_CAT_MEMORY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_GATHER,     MF_ACCESS_RANDOM,  "data", "indices", NULL, NULL) \
    MF_OP(COMPRESS,"Filter",  MF_OP_COMPRESS,MF_OP_CAT_MEMORY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, MF_ACCESS_RANDOM,  "in",   "mask", NULL, NULL) \
    MF_OP(SLICE,   "Slice",   MF_OP_SLICE,   MF_OP_CAT_MEMORY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SLICE,      MF_ACCESS_LINEAR,  "in",   "range", NULL, NULL) \
        MF_OP(RESHAPE, "Reshape", MF_OP_RESHAPE, MF_OP_CAT_MEMORY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_RESHAPE,    MF_ACCESS_LINEAR,  "in",   "shape", NULL, NULL)
    
    #endif // MF_OP_DEFS_H
    