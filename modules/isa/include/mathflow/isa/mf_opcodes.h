#ifndef MF_OPCODES_H
#define MF_OPCODES_H

// MathFlow Instruction Set Architecture
// Opcode Ranges:
// 0    - 255 : Core Math (Basic Arithmetic, Logic, Comparison)
// 256  - 511 : Array Ops (Layout, Ranges, Transformations)
// 512  - 767 : Random / Noise
// 768  - 1023: Reserved (UI / String / Etc)

#define MF_OP_LIMIT 1024

typedef enum {
    MF_OP_NOOP = 0,
    
    // --- Core Math (0 - 255) ---
    MF_OP_CORE_BEGIN = 0,

    // Arithmetic
    MF_OP_ADD = 1,
    MF_OP_SUB = 2,
    MF_OP_MUL = 3,
    MF_OP_DIV = 4,
    
    // Math Functions
    MF_OP_MIN = 10,
    MF_OP_MAX = 11,
    MF_OP_ABS = 12,
    MF_OP_CLAMP = 13,
    // MF_OP_MIX decomposed
    MF_OP_STEP = 15,
    MF_OP_SMOOTHSTEP = 16,
    
    MF_OP_FLOOR = 20,
    MF_OP_CEIL = 21,
    MF_OP_SIN = 22,
    MF_OP_COS = 23,
    MF_OP_ATAN2 = 24,
    MF_OP_SQRT = 25,
    MF_OP_POW = 26,

    // Linear Algebra
    MF_OP_MATMUL = 40,
    MF_OP_TRANSPOSE = 41,
    MF_OP_INVERSE = 42,
    MF_OP_NORMALIZE = 43,
    MF_OP_DOT = 44,
    MF_OP_LENGTH = 45,
    MF_OP_JOIN = 46, // Join two tensors into last dim: [..., 2]

    // Comparison
    MF_OP_LESS = 60,
    MF_OP_GREATER = 61,
    MF_OP_EQUAL = 62,
    MF_OP_NEQUAL = 63,
    MF_OP_LEQUAL = 64,
    MF_OP_GEQUAL = 65,

    // Logic
    MF_OP_AND = 80,
    MF_OP_OR = 81,
    MF_OP_XOR = 82,
    MF_OP_NOT = 83,

    // Control Flow / Selection
    MF_OP_WHERE_TRUE = 100,
    MF_OP_WHERE_FALSE = 101,

    MF_OP_CORE_END = 255,

    // --- Array Ops (256 - 511) ---
    MF_OP_ARRAY_BEGIN = 256,
    
    // Generators
    MF_OP_RANGE = 260, // Output: [0, 1, 2, ... N-1]
    MF_OP_INDEX = 261, // Output: Index of current element. Args: [Dest, AxisConst, -]

    // --- Array Ops ---
    MF_OP_CUMSUM = 270, // Output[i] = Sum(Input[0]...Input[i])
    
    MF_OP_COMPRESS = 280, // Output = Elements of Input where Mask is true

    MF_OP_ARRAY_END = 511,

    // --- State / Memory (512 - 767) ---
    MF_OP_STATE_BEGIN = 512,
    
    MF_OP_COPY = 520,    // Dest = Src1 (Tensor Copy)
    MF_OP_SLICE = 521,   // Dest = View of Src1[Start:End]
    MF_OP_RESHAPE = 522, // Dest = View of Src1 with new Shape

    MF_OP_STATE_END = 767,
    
    // --- Random / Noise (768 - ...) ---

} mf_opcode;

#endif // MF_OPCODES_H
