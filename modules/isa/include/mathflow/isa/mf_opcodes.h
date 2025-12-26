#ifndef MF_OPCODES_H
#define MF_OPCODES_H

typedef enum {
    MF_OP_NOOP = 0,
    
    // --- Arithmetic (Element-wise) ---
    // Works on Scalars, Vectors, and Tensors (via Broadcasting)
    MF_OP_ADD,      // C = A + B
    MF_OP_SUB,      // C = A - B
    MF_OP_MUL,      // C = A * B (Hadamard product)
    MF_OP_DIV,      // C = A / B
    
    // --- Math Functions ---
    MF_OP_MIN,
    MF_OP_MAX,
    MF_OP_ABS,
    MF_OP_CLAMP,    // Requires 3 inputs? Min/Max. Implemented as Max(Min(x, max), min)? 
                    // Or separate instruction if we stick to 2 inputs.
                    // Let's use generic Min/Max composition for Clamp.
    
    MF_OP_FLOOR,
    MF_OP_CEIL,
    MF_OP_SIN,
    MF_OP_COS,
    MF_OP_ATAN2,
    MF_OP_SQRT,
    MF_OP_POW,

    // --- Linear Algebra ---
    MF_OP_MATMUL,   // Dot Product, Matrix-Vector, Matrix-Matrix
    MF_OP_TRANSPOSE,
    MF_OP_INVERSE,
    MF_OP_NORMALIZE,

    // --- Comparison (Element-wise -> Bool Tensor) ---
    MF_OP_LESS,
    MF_OP_GREATER,
    MF_OP_EQUAL,
    MF_OP_NEQUAL,
    MF_OP_LEQUAL,
    MF_OP_GEQUAL,

    // --- Logic (Bool Tensor -> Bool Tensor) ---
    MF_OP_AND,
    MF_OP_OR,
    MF_OP_XOR,
    MF_OP_NOT,

    // --- Selection / Control Flow ---
    // Conditional Move: 
    // WHERE_TRUE:  Dest[i] = Src2[i] if Src1[i] is true
    // WHERE_FALSE: Dest[i] = Src2[i] if Src1[i] is false
    MF_OP_WHERE_TRUE,
    MF_OP_WHERE_FALSE,

    MF_OP_COUNT
} mf_opcode;

#endif // MF_OPCODES_H