#ifndef MF_OPCODES_H
#define MF_OPCODES_H

typedef enum {
    MF_OP_NOOP = 0,
    
    // --- Scalar Math (f32) ---
    MF_OP_ADD_F32,
    MF_OP_SUB_F32,
    MF_OP_MUL_F32,
    MF_OP_DIV_F32,
    MF_OP_MIN_F32,
    MF_OP_MAX_F32,
    MF_OP_CLAMP_F32,
    MF_OP_FLOOR_F32,
    MF_OP_CEIL_F32,
    
    // --- Vector Math (vec2) ---
    MF_OP_ADD_VEC2,
    MF_OP_SUB_VEC2,
    MF_OP_SCALE_VEC2,
    
    // --- Vector Math (vec3) ---
    MF_OP_ADD_VEC3,
    MF_OP_SUB_VEC3,
    MF_OP_SCALE_VEC3, // vec3 * f32
    MF_OP_DOT_VEC3,
    MF_OP_CROSS_VEC3,

    // --- Vector Math (vec4) ---
    MF_OP_ADD_VEC4,
    MF_OP_SUB_VEC4,
    MF_OP_SCALE_VEC4,
    
    // --- Matrix Math (mat4) ---
    MF_OP_MUL_MAT4,
    MF_OP_TRANS_MAT4, // Create translation matrix from vec3
    MF_OP_TRANSPOSE_MAT4,
    MF_OP_INVERSE_MAT4,

    // --- Matrix Math (mat3) ---
    MF_OP_MUL_MAT3,
    MF_OP_TRANSPOSE_MAT3,
    MF_OP_INVERSE_MAT3,
    
    // --- Comparison (f32 -> bool) ---
    // Dest: bool_col, Src1/2: f32_col
    MF_OP_LESS_F32,
    MF_OP_GREATER_F32,
    MF_OP_EQUAL_F32,

    // --- Logic (bool -> bool) ---
    MF_OP_AND,
    MF_OP_OR,
    MF_OP_NOT, // Dest = !Src1 (Src2 ignored)

    // --- Selection (bool ? src1 : src2) ---
    // Dest/Src1/Src2: Same Type, Src3 (implicit condition): bool_col
    // Note: Standard instruction is 3-addr. 
    // SELECT requires 4 operands: Dest, TrueVal, FalseVal, Cond.
    // We can reuse Src2 as Cond? No, we need both values.
    // Convention: Dest = Select(Src1, Src2). Where does Cond come from?
    // Option A: Opcode assumes Cond is in a fixed register? No.
    // Option B: Two instructions? No.
    // Option C: Use Src1 as Cond? Dest = Select(Cond, TrueVal, FalseVal)? 
    // But struct mf_instruction has only dest, src1, src2.
    // Let's redefine semantic: Dest = SELECT(Src1(Cond), Src2(True), (Implicit Next Reg?))
    // Better: use a dedicated "SELECT" logic where:
    // Dest = Src1 (if Cond) else Src2. 
    // THIS IS HARD with 3 indices.
    
    // Solution for Phase 2:
    // MF_OP_SELECT_F32: Dest = Src1, Src2 is Cond? No, where is FalseVal?
    // Let's assume for now we use "Mask" approach or simply change the instruction format?
    // Changing instruction format is risky.
    // 
    // Alternative:
    // Dest = MIX(Src1, Src2, Factor). If Factor is bool, it's Select.
    // We pack Cond into Src2? No.
    
    // Let's stick to: Dest = SELECT(Cond_Idx, True_Idx, False_Idx) ?? 
    // mf_instruction has dest, src1, src2. That's 3 slots.
    // Perfect! 
    // Dest = Result. 
    // Src1 = Condition (bool column).
    // Src2 = "Packed" index? No.
    
    // Wait, mf_instruction definition: u16 dest, u16 src1, u16 src2.
    // We need 4 operands for Select: Out, Cond, TrueVal, FalseVal.
    // 
    // Workaround:
    // 1. MF_OP_SELECT_TRUE:  Dest = Src2 if Src1(Cond) else Dest (No-op)
    // 2. MF_OP_SELECT_FALSE: Dest = Src2 if !Src1(Cond) else Dest
    // Compiler emits both to implement ternary? 
    // Dest = FalseVal
    // Dest = SelectTrue(Cond, TrueVal) -> Overwrites Dest if Cond is true.
    // This works!
    
    MF_OP_SELECT_L_F32, // Dest = Src2 (f32) if Src1 (bool) is true
    MF_OP_SELECT_L_VEC3,
    MF_OP_SELECT_L_VEC4,
    // "L" stands for "Logic" or "Latch" or just Select Left.
    // Actually, "CMOV" (Conditional Move) is the standard term.
    
    MF_OP_CMOV_TRUE_F32,  // if (bool[src1]) dest = f32[src2]
    MF_OP_CMOV_FALSE_F32, // if (!bool[src1]) dest = f32[src2]
    
    MF_OP_CMOV_TRUE_VEC3,
    MF_OP_CMOV_FALSE_VEC3,
    
    MF_OP_CMOV_TRUE_VEC4,
    MF_OP_CMOV_FALSE_VEC4,

    // --- Trigonometry (f32) ---
    MF_OP_SIN_F32,
    MF_OP_COS_F32,
    MF_OP_ATAN2_F32,

    MF_OP_COUNT
} mf_opcode;

#endif // MF_OPCODES_H
