#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/backend_cpu/mf_math.h>

// Helper macros for kernels - Updated for Accessor API
#define GET_F32(vm, idx) (mf_vm_map_f32(vm, idx))
#define GET_VEC2(vm, idx) (mf_vm_map_vec2(vm, idx))
#define GET_VEC3(vm, idx) (mf_vm_map_vec3(vm, idx))
#define GET_VEC4(vm, idx) (mf_vm_map_vec4(vm, idx))
#define GET_MAT4(vm, idx) (mf_vm_map_mat4(vm, idx))
#define GET_MAT3(vm, idx) (mf_vm_map_mat3(vm, idx))
#define GET_BOOL(vm, idx) (mf_vm_map_bool(vm, idx))

// --- Kernels ---

static void op_noop(mf_vm* vm, u16 d, u16 s1, u16 s2) { (void)vm; (void)d; (void)s1; (void)s2; }

// --- F32 Math ---
static void op_add_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = MF_VAL(s1) + MF_VAL(s2);
}

static void op_sub_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = MF_VAL(s1) - MF_VAL(s2);
}

static void op_mul_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = MF_VAL(s1) * MF_VAL(s2);
}

static void op_div_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) {
        if (MF_VAL(s2) != 0.0f) MF_VAL(d) = MF_VAL(s1) / MF_VAL(s2);
        else MF_VAL(d) = 0.0f; 
    }
}

static void op_min_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = fminf(MF_VAL(s1), MF_VAL(s2));
}

static void op_max_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = fmaxf(MF_VAL(s1), MF_VAL(s2));
}

static void op_floor_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = floorf(MF_VAL(s1));
}

static void op_ceil_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = ceilf(MF_VAL(s1));
}

static void op_sin_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = sinf(MF_VAL(s1));
}

static void op_cos_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = cosf(MF_VAL(s1));
}

static void op_atan2_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = atan2f(MF_VAL(s1), MF_VAL(s2));
}

// --- Vec3 Math ---
static void op_add_vec3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_vec3 d = GET_VEC3(vm, dest);
    mf_ref_vec3 s1 = GET_VEC3(vm, src1);
    mf_ref_vec3 s2 = GET_VEC3(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = mf_vec3_add(MF_VAL(s1), MF_VAL(s2));
}

static void op_scale_vec3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_vec3 d = GET_VEC3(vm, dest);
    mf_ref_vec3 s1 = GET_VEC3(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) {
        MF_VAL(d).x = MF_VAL(s1).x * MF_VAL(s2);
        MF_VAL(d).y = MF_VAL(s1).y * MF_VAL(s2);
        MF_VAL(d).z = MF_VAL(s1).z * MF_VAL(s2);
    }
}

static void op_dot_vec3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_vec3 s1 = GET_VEC3(vm, src1);
    mf_ref_vec3 s2 = GET_VEC3(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = mf_vec3_dot(MF_VAL(s1), MF_VAL(s2));
}

// --- Matrix ---
static void op_mul_mat4(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_mat4 d = GET_MAT4(vm, dest);
    mf_ref_mat4 s1 = GET_MAT4(vm, src1);
    mf_ref_mat4 s2 = GET_MAT4(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = mf_mat4_mul(MF_VAL(s1), MF_VAL(s2));
}

static void op_trans_mat4(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_mat4 d = GET_MAT4(vm, dest);
    mf_ref_vec3 s1 = GET_VEC3(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = mf_mat4_translate(MF_VAL(s1));
}

static void op_transpose_mat4(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_mat4 d = GET_MAT4(vm, dest);
    mf_ref_mat4 s1 = GET_MAT4(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = mf_mat4_transpose(MF_VAL(s1));
}

static void op_inverse_mat4(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_mat4 d = GET_MAT4(vm, dest);
    mf_ref_mat4 s1 = GET_MAT4(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = mf_mat4_inverse(MF_VAL(s1));
}

// --- Mat3 ---

static void op_mul_mat3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_mat3 d = GET_MAT3(vm, dest);
    mf_ref_mat3 s1 = GET_MAT3(vm, src1);
    mf_ref_mat3 s2 = GET_MAT3(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = mf_mat3_mul(MF_VAL(s1), MF_VAL(s2));
}

static void op_transpose_mat3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_mat3 d = GET_MAT3(vm, dest);
    mf_ref_mat3 s1 = GET_MAT3(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = mf_mat3_transpose(MF_VAL(s1));
}

static void op_inverse_mat3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_mat3 d = GET_MAT3(vm, dest);
    mf_ref_mat3 s1 = GET_MAT3(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = mf_mat3_inverse(MF_VAL(s1));
}

// --- Comparison ---

static void op_greater_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_bool d = GET_BOOL(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = (MF_VAL(s1) > MF_VAL(s2)) ? 1 : 0;
}

static void op_less_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_bool d = GET_BOOL(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = (MF_VAL(s1) < MF_VAL(s2)) ? 1 : 0;
}

static void op_equal_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_bool d = GET_BOOL(vm, dest);
    mf_ref_f32 s1 = GET_F32(vm, src1);
    mf_ref_f32 s2 = GET_F32(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = (MF_VAL(s1) == MF_VAL(s2)) ? 1 : 0;
}

// --- Logic ---

static void op_and(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_bool d = GET_BOOL(vm, dest);
    mf_ref_bool s1 = GET_BOOL(vm, src1);
    mf_ref_bool s2 = GET_BOOL(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = (MF_VAL(s1) && MF_VAL(s2)) ? 1 : 0;
}

static void op_or(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_ref_bool d = GET_BOOL(vm, dest);
    mf_ref_bool s1 = GET_BOOL(vm, src1);
    mf_ref_bool s2 = GET_BOOL(vm, src2);
    if (MF_VALID(d) && MF_VALID(s1) && MF_VALID(s2)) MF_VAL(d) = (MF_VAL(s1) || MF_VAL(s2)) ? 1 : 0;
}

static void op_not(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_ref_bool d = GET_BOOL(vm, dest);
    mf_ref_bool s1 = GET_BOOL(vm, src1);
    if (MF_VALID(d) && MF_VALID(s1)) MF_VAL(d) = !MF_VAL(s1);
}

// --- CMOV (Selection) ---

// F32
static void op_cmov_true_f32(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    mf_ref_bool c = GET_BOOL(vm, cond);
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s = GET_F32(vm, src);
    if (MF_VALID(c) && MF_VALID(d) && MF_VALID(s) && MF_VAL(c)) MF_VAL(d) = MF_VAL(s);
}

static void op_cmov_false_f32(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    mf_ref_bool c = GET_BOOL(vm, cond);
    mf_ref_f32 d = GET_F32(vm, dest);
    mf_ref_f32 s = GET_F32(vm, src);
    if (MF_VALID(c) && MF_VALID(d) && MF_VALID(s) && !MF_VAL(c)) MF_VAL(d) = MF_VAL(s);
}

// Vec3
static void op_cmov_true_vec3(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    mf_ref_bool c = GET_BOOL(vm, cond);
    mf_ref_vec3 d = GET_VEC3(vm, dest);
    mf_ref_vec3 s = GET_VEC3(vm, src);
    if (MF_VALID(c) && MF_VALID(d) && MF_VALID(s) && MF_VAL(c)) MF_VAL(d) = MF_VAL(s);
}

static void op_cmov_false_vec3(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    mf_ref_bool c = GET_BOOL(vm, cond);
    mf_ref_vec3 d = GET_VEC3(vm, dest);
    mf_ref_vec3 s = GET_VEC3(vm, src);
    if (MF_VALID(c) && MF_VALID(d) && MF_VALID(s) && !MF_VAL(c)) MF_VAL(d) = MF_VAL(s);
}

// Vec4
static void op_cmov_true_vec4(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    mf_ref_bool c = GET_BOOL(vm, cond);
    mf_ref_vec4 d = GET_VEC4(vm, dest);
    mf_ref_vec4 s = GET_VEC4(vm, src);
    if (MF_VALID(c) && MF_VALID(d) && MF_VALID(s) && MF_VAL(c)) MF_VAL(d) = MF_VAL(s);
}

static void op_cmov_false_vec4(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    mf_ref_bool c = GET_BOOL(vm, cond);
    mf_ref_vec4 d = GET_VEC4(vm, dest);
    mf_ref_vec4 s = GET_VEC4(vm, src);
    if (MF_VALID(c) && MF_VALID(d) && MF_VALID(s) && !MF_VAL(c)) MF_VAL(d) = MF_VAL(s);
}


// --- Init Table ---

void mf_backend_cpu_init(mf_backend_dispatch_table* table) {
    // 1. Fill default NOOPs
    for(int i=0; i<MF_OP_COUNT; ++i) table->op_table[i] = op_noop;

    // 2. Register implementations
    table->op_table[MF_OP_ADD_F32] = op_add_f32;
    table->op_table[MF_OP_SUB_F32] = op_sub_f32;
    table->op_table[MF_OP_MUL_F32] = op_mul_f32;
    table->op_table[MF_OP_DIV_F32] = op_div_f32;
    table->op_table[MF_OP_MIN_F32] = op_min_f32;
    table->op_table[MF_OP_MAX_F32] = op_max_f32;
    table->op_table[MF_OP_FLOOR_F32] = op_floor_f32;
    table->op_table[MF_OP_CEIL_F32] = op_ceil_f32;
    table->op_table[MF_OP_SIN_F32] = op_sin_f32;
    table->op_table[MF_OP_COS_F32] = op_cos_f32;
    table->op_table[MF_OP_ATAN2_F32] = op_atan2_f32;
    
    table->op_table[MF_OP_ADD_VEC3] = op_add_vec3;
    table->op_table[MF_OP_SCALE_VEC3] = op_scale_vec3;
    table->op_table[MF_OP_DOT_VEC3] = op_dot_vec3;
    
    table->op_table[MF_OP_MUL_MAT4] = op_mul_mat4;
    table->op_table[MF_OP_TRANS_MAT4] = op_trans_mat4;
    table->op_table[MF_OP_TRANSPOSE_MAT4] = op_transpose_mat4;
    table->op_table[MF_OP_INVERSE_MAT4] = op_inverse_mat4;
    
    table->op_table[MF_OP_MUL_MAT3] = op_mul_mat3;
    table->op_table[MF_OP_TRANSPOSE_MAT3] = op_transpose_mat3;
    table->op_table[MF_OP_INVERSE_MAT3] = op_inverse_mat3;
    
    // Comparison
    table->op_table[MF_OP_GREATER_F32] = op_greater_f32;
    table->op_table[MF_OP_LESS_F32] = op_less_f32;
    table->op_table[MF_OP_EQUAL_F32] = op_equal_f32;
    
    // Logic
    table->op_table[MF_OP_AND] = op_and;
    table->op_table[MF_OP_OR] = op_or;
    table->op_table[MF_OP_NOT] = op_not;
    
    // CMOV
    table->op_table[MF_OP_CMOV_TRUE_F32] = op_cmov_true_f32;
    table->op_table[MF_OP_CMOV_FALSE_F32] = op_cmov_false_f32;
    table->op_table[MF_OP_CMOV_TRUE_VEC3] = op_cmov_true_vec3;
    table->op_table[MF_OP_CMOV_FALSE_VEC3] = op_cmov_false_vec3;
    table->op_table[MF_OP_CMOV_TRUE_VEC4] = op_cmov_true_vec4;
    table->op_table[MF_OP_CMOV_FALSE_VEC4] = op_cmov_false_vec4;
}
