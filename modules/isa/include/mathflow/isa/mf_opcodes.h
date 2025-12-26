#ifndef MF_OPCODES_H
#define MF_OPCODES_H

typedef enum {
    MF_OP_NOOP = 0,
    
    // Scalar Math (f32)
    MF_OP_ADD_F32,
    MF_OP_SUB_F32,
    MF_OP_MUL_F32,
    MF_OP_DIV_F32,
    
    // Vector Math (vec3)
    MF_OP_ADD_VEC3,
    MF_OP_SUB_VEC3,
    MF_OP_SCALE_VEC3, // vec3 * f32
    MF_OP_DOT_VEC3,
    MF_OP_CROSS_VEC3,
    
    // Matrix Math (mat4)
    MF_OP_MUL_MAT4,
    MF_OP_TRANS_MAT4, // Create translation matrix from vec3
    
    MF_OP_COUNT
} mf_opcode;

#endif // MF_OPCODES_H