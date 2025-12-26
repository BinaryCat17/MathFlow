#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/backend_cpu/mf_math.h>

// Helper macros for kernels
#define GET_F32(col, idx) ((f32*)mf_column_get(col, idx))
#define GET_VEC2(col, idx) ((mf_vec2*)mf_column_get(col, idx))
#define GET_VEC3(col, idx) ((mf_vec3*)mf_column_get(col, idx))
#define GET_VEC4(col, idx) ((mf_vec4*)mf_column_get(col, idx))
#define GET_MAT4(col, idx) ((mf_mat4*)mf_column_get(col, idx))
#define GET_BOOL(col, idx) ((u8*)mf_column_get(col, idx))

// --- Kernels ---

static void op_noop(mf_vm* vm, u16 d, u16 s1, u16 s2) { (void)vm; (void)d; (void)s1; (void)s2; }

// --- F32 Math ---
static void op_add_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    f32* d = GET_F32(vm->f32_col, dest);
    f32* s1 = GET_F32(vm->f32_col, src1);
    f32* s2 = GET_F32(vm->f32_col, src2);
    if (d && s1 && s2) *d = *s1 + *s2;
}

static void op_sub_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    f32* d = GET_F32(vm->f32_col, dest);
    f32* s1 = GET_F32(vm->f32_col, src1);
    f32* s2 = GET_F32(vm->f32_col, src2);
    if (d && s1 && s2) *d = *s1 - *s2;
}

static void op_mul_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    f32* d = GET_F32(vm->f32_col, dest);
    f32* s1 = GET_F32(vm->f32_col, src1);
    f32* s2 = GET_F32(vm->f32_col, src2);
    if (d && s1 && s2) *d = *s1 * *s2;
}

// --- Vec3 Math ---
static void op_add_vec3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_vec3* d = GET_VEC3(vm->vec3_col, dest);
    mf_vec3* s1 = GET_VEC3(vm->vec3_col, src1);
    mf_vec3* s2 = GET_VEC3(vm->vec3_col, src2);
    if (d && s1 && s2) *d = mf_vec3_add(*s1, *s2);
}

static void op_scale_vec3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_vec3* d = GET_VEC3(vm->vec3_col, dest);
    mf_vec3* s1 = GET_VEC3(vm->vec3_col, src1);
    f32* s2 = GET_F32(vm->f32_col, src2);
    if (d && s1 && s2) {
        d->x = s1->x * (*s2);
        d->y = s1->y * (*s2);
        d->z = s1->z * (*s2);
    }
}

static void op_dot_vec3(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    f32* d = GET_F32(vm->f32_col, dest);
    mf_vec3* s1 = GET_VEC3(vm->vec3_col, src1);
    mf_vec3* s2 = GET_VEC3(vm->vec3_col, src2);
    if (d && s1 && s2) *d = mf_vec3_dot(*s1, *s2);
}

// --- Matrix ---
static void op_mul_mat4(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    mf_mat4* d = GET_MAT4(vm->mat4_col, dest);
    mf_mat4* s1 = GET_MAT4(vm->mat4_col, src1);
    mf_mat4* s2 = GET_MAT4(vm->mat4_col, src2);
    if (d && s1 && s2) *d = mf_mat4_mul(*s1, *s2);
}

static void op_trans_mat4(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    mf_mat4* d = GET_MAT4(vm->mat4_col, dest);
    mf_vec3* s1 = GET_VEC3(vm->vec3_col, src1);
    if (d && s1) *d = mf_mat4_translate(*s1);
}

// --- Comparison ---

static void op_greater_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    u8* d = GET_BOOL(vm->bool_col, dest);
    f32* s1 = GET_F32(vm->f32_col, src1);
    f32* s2 = GET_F32(vm->f32_col, src2);
    if (d && s1 && s2) *d = (*s1 > *s2) ? 1 : 0;
}

static void op_less_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    u8* d = GET_BOOL(vm->bool_col, dest);
    f32* s1 = GET_F32(vm->f32_col, src1);
    f32* s2 = GET_F32(vm->f32_col, src2);
    if (d && s1 && s2) *d = (*s1 < *s2) ? 1 : 0;
}

static void op_equal_f32(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    u8* d = GET_BOOL(vm->bool_col, dest);
    f32* s1 = GET_F32(vm->f32_col, src1);
    f32* s2 = GET_F32(vm->f32_col, src2);
    // Epsilon comparison is better, but strict for now
    if (d && s1 && s2) *d = (*s1 == *s2) ? 1 : 0;
}

// --- Logic ---

static void op_and(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    u8* d = GET_BOOL(vm->bool_col, dest);
    u8* s1 = GET_BOOL(vm->bool_col, src1);
    u8* s2 = GET_BOOL(vm->bool_col, src2);
    if (d && s1 && s2) *d = (*s1 && *s2) ? 1 : 0;
}

static void op_or(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    u8* d = GET_BOOL(vm->bool_col, dest);
    u8* s1 = GET_BOOL(vm->bool_col, src1);
    u8* s2 = GET_BOOL(vm->bool_col, src2);
    if (d && s1 && s2) *d = (*s1 || *s2) ? 1 : 0;
}

static void op_not(mf_vm* vm, u16 dest, u16 src1, u16 src2) {
    (void)src2;
    u8* d = GET_BOOL(vm->bool_col, dest);
    u8* s1 = GET_BOOL(vm->bool_col, src1);
    if (d && s1) *d = !(*s1);
}

// --- CMOV (Selection) ---

// F32
static void op_cmov_true_f32(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    u8* c = GET_BOOL(vm->bool_col, cond);
    f32* d = GET_F32(vm->f32_col, dest);
    f32* s = GET_F32(vm->f32_col, src);
    if (c && d && s && *c) *d = *s;
}

static void op_cmov_false_f32(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    u8* c = GET_BOOL(vm->bool_col, cond);
    f32* d = GET_F32(vm->f32_col, dest);
    f32* s = GET_F32(vm->f32_col, src);
    if (c && d && s && !(*c)) *d = *s;
}

// Vec3
static void op_cmov_true_vec3(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    u8* c = GET_BOOL(vm->bool_col, cond);
    mf_vec3* d = GET_VEC3(vm->vec3_col, dest);
    mf_vec3* s = GET_VEC3(vm->vec3_col, src);
    if (c && d && s && *c) *d = *s;
}

static void op_cmov_false_vec3(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    u8* c = GET_BOOL(vm->bool_col, cond);
    mf_vec3* d = GET_VEC3(vm->vec3_col, dest);
    mf_vec3* s = GET_VEC3(vm->vec3_col, src);
    if (c && d && s && !(*c)) *d = *s;
}

// Vec4
static void op_cmov_true_vec4(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    u8* c = GET_BOOL(vm->bool_col, cond);
    mf_vec4* d = GET_VEC4(vm->vec4_col, dest);
    mf_vec4* s = GET_VEC4(vm->vec4_col, src);
    if (c && d && s && *c) *d = *s;
}

static void op_cmov_false_vec4(mf_vm* vm, u16 dest, u16 cond, u16 src) {
    u8* c = GET_BOOL(vm->bool_col, cond);
    mf_vec4* d = GET_VEC4(vm->vec4_col, dest);
    mf_vec4* s = GET_VEC4(vm->vec4_col, src);
    if (c && d && s && !(*c)) *d = *s;
}


// --- Init Table ---

void mf_backend_cpu_init(mf_backend_dispatch_table* table) {
    // 1. Fill default NOOPs
    for(int i=0; i<MF_OP_COUNT; ++i) table->op_table[i] = op_noop;

    // 2. Register implementations
    table->op_table[MF_OP_ADD_F32] = op_add_f32;
    table->op_table[MF_OP_SUB_F32] = op_sub_f32;
    table->op_table[MF_OP_MUL_F32] = op_mul_f32;
    
    table->op_table[MF_OP_ADD_VEC3] = op_add_vec3;
    table->op_table[MF_OP_SCALE_VEC3] = op_scale_vec3;
    table->op_table[MF_OP_DOT_VEC3] = op_dot_vec3;
    
    table->op_table[MF_OP_MUL_MAT4] = op_mul_mat4;
    table->op_table[MF_OP_TRANS_MAT4] = op_trans_mat4;
    
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