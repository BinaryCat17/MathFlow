#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/vm/mf_vm_utils.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
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

// --- Kernel: SmoothStep ---
// SmoothStep(x, edges) -> edges is [min, max]
static void op_smoothstep(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* val = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* edges = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !val || !edges) return;
    
    // Edges must have last dim 2, or be size 2.
    // For MVP: Assume edges is [..., 2] and matches Val broadcasting.
    // Actually, edges might be a single Vec2 [2] (Uniform broadcast).
    
    if (!mf_utils_resolve_unary_shape(vm, dst, val)) return; // Output shape = Value shape
    dst->dtype = MF_DTYPE_F32;

    f32* X = (f32*)val->data;
    f32* E = (f32*)edges->data;
    f32* D = (f32*)dst->data;
    
    // Check if Edges is uniform (Size 2) or per-pixel
    bool uniform_edges = (edges->size == 2);
    
    for (size_t i = 0; i < dst->size; ++i) {
        float x = X[i];
        float e0 = uniform_edges ? E[0] : E[i*2 + 0];
        float e1 = uniform_edges ? E[1] : E[i*2 + 1];
        
        float t = (x - e0) / (e1 - e0);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        D[i] = t * t * (3.0f - 2.0f * t);
    }
}

// --- Kernel: GLSL Math ---

// Step(edge, x) -> 1.0 if x >= edge, else 0.0
static void op_step(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* edge = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* x = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !edge || !x) return;
    
    if (!mf_utils_resolve_binary_shape(vm, dst, edge, x)) return;
    // Output is F32 (0.0 or 1.0)
    dst->dtype = MF_DTYPE_F32;
    
    f32* de = (f32*)edge->data; f32* dx = (f32*)x->data; f32* dd = (f32*)dst->data;
    bool e_s = (edge->size == 1); bool x_s = (x->size == 1);
    
    for(size_t i=0; i<dst->size; ++i) {
        f32 e_val = e_s ? de[0] : de[i];
        f32 x_val = x_s ? dx[0] : dx[i];
        dd[i] = (x_val >= e_val) ? 1.0f : 0.0f;
    }
}

// Dot(a, b) -> Sum(a*b) along last axis
static void op_dot(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    
    if (a->size != b->size) return; // Strict size check for now

    // Determine output shape: A without last dim
    int out_ndim = (a->ndim > 0) ? a->ndim - 1 : 0;
    
    // Resize dst (copies shape from A, but truncated)
    if (!mf_vm_resize_tensor(vm, dst, a->shape, out_ndim)) return;
    dst->dtype = MF_DTYPE_F32;

    f32* A = (f32*)a->data; 
    f32* B = (f32*)b->data; 
    f32* D = (f32*)dst->data;
    
    size_t dim = (a->ndim <= 1) ? a->size : a->shape[a->ndim-1];
    size_t batch = a->size / dim;
    
    for (size_t i = 0; i < batch; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            sum += A[i*dim + k] * B[i*dim + k];
        }
        D[i] = sum;
    }
}

// Length(a) -> Sqrt(Dot(a, a))
static void op_length(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;

    int out_ndim = (a->ndim > 0) ? a->ndim - 1 : 0;
    if (!mf_vm_resize_tensor(vm, dst, a->shape, out_ndim)) return;
    dst->dtype = MF_DTYPE_F32;

    f32* A = (f32*)a->data; 
    f32* D = (f32*)dst->data;
    
    size_t dim = (a->ndim <= 1) ? a->size : a->shape[a->ndim-1];
    size_t batch = a->size / dim;
    
    for (size_t i = 0; i < batch; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            float val = A[i*dim + k];
            sum += val * val;
        }
        D[i] = sqrtf(sum);
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

    // Fast Path
    if (dim == 4 && a->size == 16) {
        mf_mat4 A, B; 
        memcpy(A.m, a->data, sizeof(mf_mat4));
        memcpy(B.m, b->data, sizeof(mf_mat4));
        mf_mat4 R = mf_mat4_mul(A, B);
        memcpy(dst->data, R.m, sizeof(mf_mat4));
        return;
    }
    if (dim == 3 && a->size == 9) {
        mf_mat3 A, B; 
        memcpy(A.m, a->data, sizeof(mf_mat3));
        memcpy(B.m, b->data, sizeof(mf_mat3));
        mf_mat3 R = mf_mat3_mul(A, B);
        memcpy(dst->data, R.m, sizeof(mf_mat3));
        return;
    }

    // Generic Path
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
    
    int dim = (int)sqrtf((float)a->size);
    
    // Fast Path
    if (dim == 4 && a->size == 16) {
        mf_mat4 A; memcpy(A.m, a->data, sizeof(mf_mat4));
        mf_mat4 R = mf_mat4_transpose(A);
        memcpy(dst->data, R.m, sizeof(mf_mat4));
        return;
    }
    if (dim == 3 && a->size == 9) {
        mf_mat3 A; memcpy(A.m, a->data, sizeof(mf_mat3));
        mf_mat3 R = mf_mat3_transpose(A);
        memcpy(dst->data, R.m, sizeof(mf_mat3));
        return;
    }

    // Generic Path
    f32* src = (f32*)a->data; f32* out = (f32*)dst->data;
    for (int r = 0; r < dim; r++) for (int c = 0; c < dim; c++) out[c * dim + r] = src[r * dim + c];
}

static void op_inverse(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;
    
    dst->dtype = a->dtype;
    if (!mf_vm_resize_tensor(vm, dst, a->shape, a->ndim)) return;

    int dim = (int)sqrtf((float)a->size);
    
    if (dim == 3 && a->size == 9) {
        mf_mat3 m;
        memcpy(m.m, a->data, sizeof(mf_mat3));
        mf_mat3 res = mf_mat3_inverse(m);
        memcpy(dst->data, res.m, sizeof(mf_mat3));
    } 
    else if (dim == 4 && a->size == 16) {
        mf_mat4 m;
        memcpy(m.m, a->data, sizeof(mf_mat4));
        mf_mat4 res = mf_mat4_inverse(m);
        memcpy(dst->data, res.m, sizeof(mf_mat4));
    }
    else {
        // Fallback: Identity / Copy (Not implemented generic inverse)
        memcpy(dst->data, a->data, a->size * sizeof(f32));
    }
}

// --- Kernel: Join (Pack Vec2) ---
// Join(a, b) -> [..., 2] where ... is the common shape
static void op_join(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;

    // Resolve common shape (broadcasting rules apply for inputs)
    // But output will have +1 dimension
    
    // For now, simplify: Assume strict equality or scalar broadcast
    // Target shape logic:
    // If A is [H, W], B is [H, W] -> Dst is [H, W, 2]
    // If A is [W], B is [W] -> Dst is [W, 2]
    
    // We can reuse resolve_binary_shape to find the "base" shape, then append 2
    
    // Manual broadcasting logic simplified for Join
    // 1. Determine Output NDIM
    //    max(a->ndim, b->ndim) + 1
    //    limit checks...
    
    // Let's rely on standard broadcasting to determine the "element-wise" shape first
    // This is a bit tricky because we can't use the standard helper directly on DST because DST has different rank.
    // Hack: Create a fake tensor struct for shape resolution
    
    // Simplified implementation: Strict shape match required for Phase 9 MVP (except scalar)
    size_t size = 1;
    if (a->size != b->size) {
        // Allow scalar broadcast? Not for Join usually, but maybe. 
        // If A is [10], B is [1], result [10, 2]? Yes.
        // For MVP: Require sizes to match.
        // TODO: Implement proper broadcasting for Join
        if (a->size != b->size) return; 
    }
    size = a->size;
    
    // Setup Output Shape
    for (int i=0; i<a->ndim; ++i) dst->shape[i] = a->shape[i];
    dst->shape[a->ndim] = 2;
    dst->ndim = a->ndim + 1;
    dst->size = size * 2;
    dst->dtype = a->dtype; // Assume same type

    // Realloc if needed
    // NOTE: mf_vm_resize_tensor expects the shape to be passed.
    if (!mf_vm_resize_tensor(vm, dst, dst->shape, dst->ndim)) return;
    
    f32* A = (f32*)a->data; 
    f32* B = (f32*)b->data; 
    f32* D = (f32*)dst->data;
    
    for (size_t i = 0; i < size; ++i) {
        D[i*2 + 0] = A[i];
        D[i*2 + 1] = B[i];
    }
}

// --- Kernel: Memory/State ---
static void op_copy(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !src) return;

    dst->dtype = src->dtype;
    if (!mf_vm_resize_tensor(vm, dst, src->shape, src->ndim)) return;

    size_t size = mf_dtype_size(src->dtype) * src->size;
    memcpy(dst->data, src->data, size);
}

// --- Registration ---
void mf_ops_core_register(mf_backend_dispatch_table* table) {
    // Core Math
    table->op_table[MF_OP_ADD] = op_add; table->op_table[MF_OP_SUB] = op_sub; table->op_table[MF_OP_MUL] = op_mul; table->op_table[MF_OP_DIV] = op_div;
    table->op_table[MF_OP_SIN] = op_sin; table->op_table[MF_OP_COS] = op_cos; table->op_table[MF_OP_FLOOR] = op_floor; table->op_table[MF_OP_CEIL] = op_ceil;
    table->op_table[MF_OP_ABS] = op_abs; table->op_table[MF_OP_SQRT] = op_sqrt; table->op_table[MF_OP_ATAN2] = op_atan2; table->op_table[MF_OP_POW] = op_pow;
    table->op_table[MF_OP_MIN] = op_min; table->op_table[MF_OP_MAX] = op_max;
    table->op_table[MF_OP_SMOOTHSTEP] = op_smoothstep;
    // GLSL Math
    table->op_table[MF_OP_STEP] = op_step;
    table->op_table[MF_OP_DOT] = op_dot;
    table->op_table[MF_OP_LENGTH] = op_length;

    // Comparison
    table->op_table[MF_OP_LESS] = op_less; table->op_table[MF_OP_GREATER] = op_greater; table->op_table[MF_OP_EQUAL] = op_equal;
    table->op_table[MF_OP_NEQUAL] = op_nequal; table->op_table[MF_OP_LEQUAL] = op_lequal; table->op_table[MF_OP_GEQUAL] = op_gequal;
    // Logic
    table->op_table[MF_OP_AND] = op_and; table->op_table[MF_OP_OR] = op_or; table->op_table[MF_OP_NOT] = op_not;
    // Selection
    table->op_table[MF_OP_WHERE_TRUE] = op_where_true; table->op_table[MF_OP_WHERE_FALSE] = op_where_false;
    // Matrix
    table->op_table[MF_OP_MATMUL] = op_matmul; table->op_table[MF_OP_TRANSPOSE] = op_transpose; table->op_table[MF_OP_INVERSE] = op_inverse;
    table->op_table[MF_OP_JOIN] = op_join;
    // State
    table->op_table[MF_OP_COPY] = op_copy;
}