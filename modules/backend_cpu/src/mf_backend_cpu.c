#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/vm/mf_vm.h> // For mf_vm_resize_tensor
#include <string.h>
#include <math.h>

// --- Helper: Shape Resolution ---

// Simple broadcasting rule: Output takes the shape of the larger tensor.
// True broadcasting (NumPy style) would be more complex, but this fits our current needs.
static bool resolve_binary_shape(mf_vm* vm, mf_tensor* dst, const mf_tensor* a, const mf_tensor* b) {
    const mf_tensor* shape_src = (a->size >= b->size) ? a : b;
    // Inherit dtype from 'a' usually, but here we assume inputs are compatible floats
    // or the op defines the output type (like compare -> u8).
    // The caller should set dst->dtype if it differs (e.g. comparison).
    
    // Default: output dtype matches A (usually f32)
    if (dst->dtype == MF_DTYPE_UNKNOWN) dst->dtype = a->dtype;

    return mf_vm_resize_tensor(vm, dst, shape_src->shape, shape_src->ndim);
}

static bool resolve_unary_shape(mf_vm* vm, mf_tensor* dst, const mf_tensor* a) {
    if (dst->dtype == MF_DTYPE_UNKNOWN) dst->dtype = a->dtype;
    return mf_vm_resize_tensor(vm, dst, a->shape, a->ndim);
}

// --- Kernel: Binary Math ---
#define OP_KERNEL_BINARY(NAME, OP) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!resolve_binary_shape(vm, dst, a, b)) return; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); \
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = (a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i]); \
}
OP_KERNEL_BINARY(add, +)
OP_KERNEL_BINARY(sub, -)
OP_KERNEL_BINARY(mul, *)
OP_KERNEL_BINARY(div, /)

#define OP_KERNEL_BINARY_FUNC(NAME, FUNC) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!resolve_binary_shape(vm, dst, a, b)) return; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); \
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = FUNC((a_s ? da[0] : da[i]), (b_s ? db[0] : db[i])); \
}
OP_KERNEL_BINARY_FUNC(atan2, atan2f)
OP_KERNEL_BINARY_FUNC(pow, powf)

// --- Kernel: Unary Math ---
#define OP_KERNEL_UNARY(NAME, FUNC) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    if (!dst || !a) return; \
    if (!resolve_unary_shape(vm, dst, a)) return; \
    f32* da = (f32*)a->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = FUNC(da[i]); \
}
OP_KERNEL_UNARY(sin, sinf)
OP_KERNEL_UNARY(cos, cosf)
OP_KERNEL_UNARY(floor, floorf)
OP_KERNEL_UNARY(ceil, ceilf)
OP_KERNEL_UNARY(abs, fabsf)
OP_KERNEL_UNARY(sqrt, sqrtf)

// --- Kernel: Binary Min/Max ---
static void op_min(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    if (!resolve_binary_shape(vm, dst, a, b)) return;
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data;
    bool a_s = (a->size == 1); bool b_s = (b->size == 1);
    for(size_t i=0; i<dst->size; ++i) { f32 va = a_s ? da[0] : da[i]; f32 vb = b_s ? db[0] : db[i]; dd[i] = (va < vb) ? va : vb; }
}
static void op_max(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    if (!resolve_binary_shape(vm, dst, a, b)) return;
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data;
    bool a_s = (a->size == 1); bool b_s = (b->size == 1);
    for(size_t i=0; i<dst->size; ++i) { f32 va = a_s ? da[0] : da[i]; f32 vb = b_s ? db[0] : db[i]; dd[i] = (va > vb) ? va : vb; }
}

// --- Kernel: Comparison ---
#define OP_KERNEL_COMPARE(NAME, OP) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->dtype = MF_DTYPE_U8; /* Force Bool Output */ \
    if (!resolve_binary_shape(vm, dst, a, b)) return; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); u8* dd = (u8*)dst->data; \
    if (a->dtype == MF_DTYPE_F32) { \
        f32* da = (f32*)a->data; f32* db = (f32*)b->data; \
        for(size_t i=0; i<dst->size; ++i) dd[i] = ((a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i])) ? 1 : 0; \
    } else if (a->dtype == MF_DTYPE_I32) { \
        int32_t* da = (int32_t*)a->data; int32_t* db = (int32_t*)b->data; \
        for(size_t i=0; i<dst->size; ++i) dd[i] = ((a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i])) ? 1 : 0; \
    } \
}
OP_KERNEL_COMPARE(less, <)
OP_KERNEL_COMPARE(greater, >)
OP_KERNEL_COMPARE(equal, ==)
OP_KERNEL_COMPARE(nequal, !=)
OP_KERNEL_COMPARE(lequal, <=)
OP_KERNEL_COMPARE(gequal, >=)

// --- Kernel: Logic ---
#define OP_KERNEL_LOGIC(NAME, OP) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->dtype = MF_DTYPE_U8; \
    if (!resolve_binary_shape(vm, dst, a, b)) return; \
    u8* da = (u8*)a->data; u8* db = (u8*)b->data; u8* dd = (u8*)dst->data; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); \
    for(size_t i=0; i<dst->size; ++i) dd[i] = (a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i]); \
}
OP_KERNEL_LOGIC(and, &&)
OP_KERNEL_LOGIC(or, ||)
static void op_not(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return; 
    dst->dtype = MF_DTYPE_U8;
    if (!resolve_unary_shape(vm, dst, a)) return;
    u8* da = (u8*)a->data; u8* dd = (u8*)dst->data;
    for(size_t i=0; i<dst->size; ++i) dd[i] = !da[i];
}

// --- Kernel: Where (Select) ---
static void op_where_true(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* val = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !cond || !val) return;
    
    // Output shape matches the larger of (cond, val)
    if (!resolve_binary_shape(vm, dst, cond, val)) return;
    dst->dtype = val->dtype; // Enforce type
    
    u8* c = (u8*)cond->data; u8* v = (u8*)val->data; u8* d = (u8*)dst->data; size_t es = mf_dtype_size(val->dtype);
    bool c_s = (cond->size == 1); bool v_s = (val->size == 1);
    for(size_t i=0; i<dst->size; ++i) if (c[c_s ? 0 : i]) memcpy(d + i*es, v + (v_s ? 0 : i*es), es);
}
static void op_where_false(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* val = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !cond || !val) return;

    if (!resolve_binary_shape(vm, dst, cond, val)) return;
    dst->dtype = val->dtype;

    u8* c = (u8*)cond->data; u8* v = (u8*)val->data; u8* d = (u8*)dst->data; size_t es = mf_dtype_size(val->dtype);
    bool c_s = (cond->size == 1); bool v_s = (val->size == 1);
    for(size_t i=0; i<dst->size; ++i) if (!c[c_s ? 0 : i]) memcpy(d + i*es, v + (v_s ? 0 : i*es), es);
}

// --- Kernel: Matrix ---
static void op_matmul(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    
    // Validation
    int dim = (int)sqrtf((float)a->size); 
    if (dim * dim != a->size) return; // Only square matrices for now
    
    dst->dtype = a->dtype;
    if (!mf_vm_resize_tensor(vm, dst, a->shape, a->ndim)) return;

    f32* A = (f32*)a->data; f32* B = (f32*)b->data; f32* C = (f32*)dst->data;
    for (int r = 0; r < dim; r++) for (int c = 0; c < dim; c++) { 
        float sum = 0.0f; 
        for (int k = 0; k < dim; k++) sum += A[r * dim + k] * B[k * dim + c]; 
        C[r * dim + c] = sum; 
    }
}
static void op_transpose(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return; 
    
    // Transpose works by swapping dims. For 2D square only now.
    // TODO: Generic transpose
    dst->dtype = a->dtype;
    if (!mf_vm_resize_tensor(vm, dst, a->shape, a->ndim)) return;
    
    int dim = (int)sqrtf((float)a->size); f32* src = (f32*)a->data; f32* out = (f32*)dst->data;
    for (int r = 0; r < dim; r++) for (int c = 0; c < dim; c++) out[c * dim + r] = src[r * dim + c];
}
static void op_inverse(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;
    
    dst->dtype = a->dtype;
    if (!mf_vm_resize_tensor(vm, dst, a->shape, a->ndim)) return;

    f32* m = (f32*)a->data; f32* out = (f32*)dst->data; int dim = (int)sqrtf((float)a->size);
    if (dim == 3) {
        float det = m[0]*(m[4]*m[8]-m[7]*m[5]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]); float invDet = 1.0f/det;
        out[0]=(m[4]*m[8]-m[5]*m[7])*invDet; out[1]=(m[2]*m[7]-m[1]*m[8])*invDet; out[2]=(m[1]*m[5]-m[2]*m[4])*invDet;
        out[3]=(m[5]*m[6]-m[3]*m[8])*invDet; out[4]=(m[0]*m[8]-m[2]*m[6])*invDet; out[5]=(m[2]*m[3]-m[0]*m[5])*invDet;
        out[6]=(m[3]*m[7]-m[4]*m[6])*invDet; out[7]=(m[1]*m[6]-m[0]*m[7])*invDet; out[8]=(m[0]*m[4]-m[1]*m[3])*invDet;
    } else memcpy(out, m, a->size * sizeof(f32));
}

void mf_backend_cpu_init(mf_backend_dispatch_table* table) {
    memset(table, 0, sizeof(mf_backend_dispatch_table));
    table->op_table[MF_OP_ADD] = op_add; table->op_table[MF_OP_SUB] = op_sub; table->op_table[MF_OP_MUL] = op_mul; table->op_table[MF_OP_DIV] = op_div;
    table->op_table[MF_OP_SIN] = op_sin; table->op_table[MF_OP_COS] = op_cos; table->op_table[MF_OP_FLOOR] = op_floor; table->op_table[MF_OP_CEIL] = op_ceil;
    table->op_table[MF_OP_ABS] = op_abs; table->op_table[MF_OP_SQRT] = op_sqrt; table->op_table[MF_OP_ATAN2] = op_atan2; table->op_table[MF_OP_POW] = op_pow;
    table->op_table[MF_OP_MIN] = op_min; table->op_table[MF_OP_MAX] = op_max;
    table->op_table[MF_OP_LESS] = op_less; table->op_table[MF_OP_GREATER] = op_greater; table->op_table[MF_OP_EQUAL] = op_equal;
    table->op_table[MF_OP_NEQUAL] = op_nequal; table->op_table[MF_OP_LEQUAL] = op_lequal; table->op_table[MF_OP_GEQUAL] = op_gequal;
    table->op_table[MF_OP_AND] = op_and; table->op_table[MF_OP_OR] = op_or; table->op_table[MF_OP_NOT] = op_not;
    table->op_table[MF_OP_WHERE_TRUE] = op_where_true; table->op_table[MF_OP_WHERE_FALSE] = op_where_false;
    table->op_table[MF_OP_MATMUL] = op_matmul; table->op_table[MF_OP_TRANSPOSE] = op_transpose; table->op_table[MF_OP_INVERSE] = op_inverse;
}
