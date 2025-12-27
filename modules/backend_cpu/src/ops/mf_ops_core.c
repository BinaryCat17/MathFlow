#include <mathflow/backend_cpu/ops/mf_ops_core.h>
#include <mathflow/backend_cpu/mf_backend_utils.h>
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>
#include <math.h>

// --- Kernel: Binary Math ---
MF_KERNEL_BINARY(add, +)
MF_KERNEL_BINARY(sub, -)
MF_KERNEL_BINARY(mul, *)
MF_KERNEL_BINARY(div, /)

MF_KERNEL_BINARY_FUNC(atan2, atan2f)
MF_KERNEL_BINARY_FUNC(pow, powf)

// --- Kernel: Unary Math ---
MF_KERNEL_UNARY(sin, sinf)
MF_KERNEL_UNARY(cos, cosf)
MF_KERNEL_UNARY(floor, floorf)
MF_KERNEL_UNARY(ceil, ceilf)
MF_KERNEL_UNARY(abs, fabsf)
MF_KERNEL_UNARY(sqrt, sqrtf)

// --- Kernel: Binary Min/Max ---
static void op_min(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    if (!mf_utils_resolve_binary_shape(vm, dst, a, b)) return;
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data;
    bool a_s = (a->size == 1); bool b_s = (b->size == 1);
    for(size_t i=0; i<dst->size; ++i) { 
        f32 va = a_s ? da[0] : da[i]; 
        f32 vb = b_s ? db[0] : db[i]; 
        dd[i] = (va < vb) ? va : vb; 
    }
}

static void op_max(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    if (!mf_utils_resolve_binary_shape(vm, dst, a, b)) return;
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data;
    bool a_s = (a->size == 1); bool b_s = (b->size == 1);
    for(size_t i=0; i<dst->size; ++i) { 
        f32 va = a_s ? da[0] : da[i]; 
        f32 vb = b_s ? db[0] : db[i]; 
        dd[i] = (va > vb) ? va : vb; 
    }
}

// --- Kernel: Comparison ---
MF_KERNEL_COMPARE(less, <)
MF_KERNEL_COMPARE(greater, >)
MF_KERNEL_COMPARE(equal, ==)
MF_KERNEL_COMPARE(nequal, !=)
MF_KERNEL_COMPARE(lequal, <=)
MF_KERNEL_COMPARE(gequal, >=)

// --- Kernel: Logic ---
MF_KERNEL_LOGIC(and, &&)
MF_KERNEL_LOGIC(or, ||)

static void op_not(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return; 
    dst->dtype = MF_DTYPE_U8;
    if (!mf_utils_resolve_unary_shape(vm, dst, a)) return;
    u8* da = (u8*)a->data; u8* dd = (u8*)dst->data;
    for(size_t i=0; i<dst->size; ++i) dd[i] = !da[i];
}

// --- Kernel: Where (Select) ---
static void op_where_true(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* val = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !cond || !val) return;
    
    if (!mf_utils_resolve_binary_shape(vm, dst, cond, val)) return;
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

    if (!mf_utils_resolve_binary_shape(vm, dst, cond, val)) return;
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
    
    int dim = (int)sqrtf((float)a->size); 
    if (dim * dim != a->size) return; 
    
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

// --- Registration ---
void mf_ops_core_register(mf_backend_dispatch_table* table) {
    // Core Math
    table->op_table[MF_OP_ADD] = op_add; table->op_table[MF_OP_SUB] = op_sub; table->op_table[MF_OP_MUL] = op_mul; table->op_table[MF_OP_DIV] = op_div;
    table->op_table[MF_OP_SIN] = op_sin; table->op_table[MF_OP_COS] = op_cos; table->op_table[MF_OP_FLOOR] = op_floor; table->op_table[MF_OP_CEIL] = op_ceil;
    table->op_table[MF_OP_ABS] = op_abs; table->op_table[MF_OP_SQRT] = op_sqrt; table->op_table[MF_OP_ATAN2] = op_atan2; table->op_table[MF_OP_POW] = op_pow;
    table->op_table[MF_OP_MIN] = op_min; table->op_table[MF_OP_MAX] = op_max;
    // Comparison
    table->op_table[MF_OP_LESS] = op_less; table->op_table[MF_OP_GREATER] = op_greater; table->op_table[MF_OP_EQUAL] = op_equal;
    table->op_table[MF_OP_NEQUAL] = op_nequal; table->op_table[MF_OP_LEQUAL] = op_lequal; table->op_table[MF_OP_GEQUAL] = op_gequal;
    // Logic
    table->op_table[MF_OP_AND] = op_and; table->op_table[MF_OP_OR] = op_or; table->op_table[MF_OP_NOT] = op_not;
    // Selection
    table->op_table[MF_OP_WHERE_TRUE] = op_where_true; table->op_table[MF_OP_WHERE_FALSE] = op_where_false;
    // Matrix
    table->op_table[MF_OP_MATMUL] = op_matmul; table->op_table[MF_OP_TRANSPOSE] = op_transpose; table->op_table[MF_OP_INVERSE] = op_inverse;
}
