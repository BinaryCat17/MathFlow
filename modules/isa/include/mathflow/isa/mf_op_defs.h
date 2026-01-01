#ifndef MF_OP_DEFS_H
#define MF_OP_DEFS_H

/**
 * MathFlow Operation Definitions (Single Source of Truth)
 * 
 * Format:
 * MF_OP(node_suffix, json_name, opcode, category, type_mask, type_rule, shape_rule, p1, p2, p3)
 * 
 * Categories:
 * - SPECIAL: Handled by compiler/host, may not emit direct opcode (Const, Input, Call)
 * - UNARY: 1 input -> 1 output
 * - BINARY: 2 inputs -> 1 output
 * - TERNARY: 3 inputs -> 1 output
 * - MATRIX: Matrix specific operations
 * - ARRAY: Array/Generator operations
 */

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
    MF_SHAPE_BROADCAST,     // Broadcast S1 and S2 (and S3 if present)
    MF_SHAPE_MATMUL,        // Matrix Multiplication [M,K] x [K,N] -> [M,N]
    MF_SHAPE_TRANSPOSE,     // Swap dim 0 and 1
    MF_SHAPE_DOT,           // Dot product (reduces last dim)
    MF_SHAPE_JOIN,          // Join/Concat (adds dimension)
    MF_SHAPE_GATHER,        // Shape follows indices
    MF_SHAPE_RESHAPE,       // Shape follows constant value
    MF_SHAPE_SLICE,         // 1D slice
    MF_SHAPE_DYNAMIC_1D,    // For Index/Range
} mf_shape_rule;

#define MF_OP_LIST \
    /* --- Special Nodes --- */ \
    MF_OP(CONST,   "Const",   MF_OP_NOOP,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SPECIAL,    "out", NULL,  NULL) \
    MF_OP(INPUT,   "Input",   MF_OP_NOOP,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SPECIAL,    "out", NULL,  NULL) \
    MF_OP(OUTPUT,  "Output",  MF_OP_COPY,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, "in",  NULL,  NULL) \
    MF_OP(CALL,    "Call",    MF_OP_NOOP,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SPECIAL,    NULL,  NULL,  NULL) \
    \
    /* --- Core Math: Binary --- */ \
    MF_OP(ADD,     "Add",     MF_OP_ADD,     MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(SUB,     "Sub",     MF_OP_SUB,     MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(MUL,     "Mul",     MF_OP_MUL,     MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(DIV,     "Div",     MF_OP_DIV,     MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(POW,     "Pow",     MF_OP_POW,     MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "base","exp", NULL) \
    MF_OP(ATAN2,   "Atan2",   MF_OP_ATAN2,   MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "y",   "x",   NULL) \
    MF_OP(MIN,     "Min",     MF_OP_MIN,     MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(MAX,     "Max",     MF_OP_MAX,     MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    \
    /* --- Core Math: Unary --- */ \
    MF_OP(ABS,     "Abs",     MF_OP_ABS,     MF_OP_CAT_UNARY,   MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, "x",   NULL,  NULL) \
    MF_OP(SIN,     "Sin",     MF_OP_SIN,     MF_OP_CAT_UNARY,   MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "x",   NULL,  NULL) \
    MF_OP(COS,     "Cos",     MF_OP_COS,     MF_OP_CAT_UNARY,   MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "x",   NULL,  NULL) \
    MF_OP(SQRT,    "Sqrt",    MF_OP_SQRT,    MF_OP_CAT_UNARY,   MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "x",   NULL,  NULL) \
    MF_OP(FLOOR,   "Floor",   MF_OP_FLOOR,   MF_OP_CAT_UNARY,   MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "x",   NULL,  NULL) \
    MF_OP(CEIL,    "Ceil",    MF_OP_CEIL,    MF_OP_CAT_UNARY,   MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "x",   NULL,  NULL) \
    MF_OP(NOT,     "Not",     MF_OP_NOT,     MF_OP_CAT_UNARY,   MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_SAME_AS_S1, "in",  NULL,  NULL) \
    \
    /* --- Core Math: Ternary --- */ \
    MF_OP(SELECT,  "Select",  MF_OP_SELECT,  MF_OP_CAT_TERNARY, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT_2, MF_SHAPE_BROADCAST,"cond","true","false") \
    MF_OP(MIX,     "Mix",     MF_OP_MIX,     MF_OP_CAT_TERNARY, MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_BROADCAST,  "a",   "b",   "t") \
    MF_OP(CLAMP,   "Clamp",   MF_OP_CLAMP,   MF_OP_CAT_TERNARY, MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "x",   "min", "max") \
    MF_OP(STEP,    "Step",    MF_OP_STEP,    MF_OP_CAT_BINARY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_BROADCAST,  "edge","x",   NULL) \
    MF_OP(SMOOTHSTEP, "SmoothStep", MF_OP_SMOOTHSTEP, MF_OP_CAT_BINARY, MF_TYPE_MASK_F32, MF_OUT_FORCE_F32, MF_SHAPE_BROADCAST,  "edges", "x", NULL) \
    \
    /* --- Matrix --- */ \
    MF_OP(MATMUL,  "MatMul",    MF_OP_MATMUL,    MF_OP_CAT_MATRIX, MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_MATMUL,    "a",   "b",   NULL) \
    MF_OP(TRANSPOSE,"Transpose", MF_OP_TRANSPOSE, MF_OP_CAT_MATRIX, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_TRANSPOSE, "in",  NULL,  NULL) \
    MF_OP(INVERSE, "Inverse",   MF_OP_INVERSE,   MF_OP_CAT_MATRIX, MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "in",  NULL,  NULL) \
    MF_OP(NORMALIZE,"Normalize", MF_OP_NORMALIZE, MF_OP_CAT_UNARY,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "in",  NULL,  NULL) \
    MF_OP(DOT,     "Dot",       MF_OP_DOT,       MF_OP_CAT_MATRIX, MF_TYPE_MASK_NUMERIC, MF_OUT_FORCE_F32,     MF_SHAPE_DOT,       "a",   "b",   NULL) \
    MF_OP(LENGTH,  "Length",    MF_OP_LENGTH,    MF_OP_CAT_UNARY,  MF_TYPE_MASK_F32,     MF_OUT_FORCE_F32,     MF_SHAPE_SAME_AS_S1, "x",   NULL,  NULL) \
    MF_OP(JOIN,    "Join",      MF_OP_JOIN,      MF_OP_CAT_MATRIX, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_JOIN,      "a",   "b",   NULL) \
    \
    /* --- Comparison --- */ \
    MF_OP(LESS,    "Less",    MF_OP_LESS,    MF_OP_CAT_BINARY, MF_TYPE_MASK_NUMERIC, MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(GREATER, "Greater", MF_OP_GREATER, MF_OP_CAT_BINARY, MF_TYPE_MASK_NUMERIC, MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(EQUAL,   "Equal",   MF_OP_EQUAL,   MF_OP_CAT_BINARY, MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(NEQUAL,  "NotEqual",MF_OP_NEQUAL,  MF_OP_CAT_BINARY, MF_TYPE_MASK_ALL,     MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(LEQUAL,  "LessEqual",MF_OP_LEQUAL, MF_OP_CAT_BINARY, MF_TYPE_MASK_NUMERIC, MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(GEQUAL,  "GreaterEqual",MF_OP_GEQUAL, MF_OP_CAT_BINARY, MF_TYPE_MASK_NUMERIC, MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    \
    /* --- Logic --- */ \
    MF_OP(AND,     "And",     MF_OP_AND,     MF_OP_CAT_BINARY, MF_TYPE_MASK_LOGIC,   MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(OR,      "Or",      MF_OP_OR,      MF_OP_CAT_BINARY, MF_TYPE_MASK_LOGIC,   MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    MF_OP(XOR,     "Xor",     MF_OP_XOR,     MF_OP_CAT_BINARY, MF_TYPE_MASK_LOGIC,   MF_OUT_FORCE_U8,      MF_SHAPE_BROADCAST,  "a",   "b",   NULL) \
    \
    /* --- Array Ops --- */ \
    MF_OP(RANGE,   "Range",   MF_OP_RANGE,   MF_OP_CAT_ARRAY,  MF_TYPE_MASK_NUMERIC, MF_OUT_FORCE_F32,     MF_SHAPE_DYNAMIC_1D, "count",NULL, NULL) \
    MF_OP(INDEX,   "Index",   MF_OP_INDEX,   MF_OP_CAT_ARRAY,  MF_TYPE_MASK_NUMERIC, MF_OUT_FORCE_F32,     MF_SHAPE_DYNAMIC_1D, "axis", NULL, NULL) \
    MF_OP(GATHER,  "Gather",  MF_OP_GATHER,  MF_OP_CAT_ARRAY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_GATHER,     "data", "indices", NULL) \
    MF_OP(CUMSUM,  "CumSum",  MF_OP_CUMSUM,  MF_OP_CAT_ARRAY,  MF_TYPE_MASK_NUMERIC, MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, "in",   NULL, NULL) \
    MF_OP(COMPRESS,"Filter",  MF_OP_COMPRESS,MF_OP_CAT_ARRAY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, "in",   "mask", NULL) \
    MF_OP(SLICE,   "Slice",   MF_OP_SLICE,   MF_OP_CAT_ARRAY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SLICE,      "in",   "range", NULL) \
    MF_OP(RESHAPE, "Reshape", MF_OP_RESHAPE, MF_OP_CAT_ARRAY,  MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_RESHAPE,    "in",   "shape", NULL) \
    MF_OP(COPY,    "Copy",    MF_OP_COPY,    MF_OP_CAT_SPECIAL, MF_TYPE_MASK_ALL,     MF_OUT_SAME_AS_INPUT, MF_SHAPE_SAME_AS_S1, "in",  NULL, NULL)


typedef enum {
    MF_OP_CAT_SPECIAL,
    MF_OP_CAT_UNARY,
    MF_OP_CAT_BINARY,
    MF_OP_CAT_TERNARY,
    MF_OP_CAT_MATRIX,
    MF_OP_CAT_ARRAY,
} mf_op_category;

#endif // MF_OP_DEFS_H