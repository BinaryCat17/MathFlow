#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>
#include <math.h>

// Dot(a, b) -> Sum(a*b) along last axis
static void op_dot(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    
    if (a->size != b->size) return; 

    int out_ndim = (a->ndim > 0) ? a->ndim - 1 : 0;
    
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->shape, out_ndim)) return;
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
static void op_length(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;

    int out_ndim = (a->ndim > 0) ? a->ndim - 1 : 0;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->shape, out_ndim)) return;
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

static void op_matmul(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    
    int dim = (int)sqrtf((float)a->size); 
    if (dim * dim != a->size) return; 
    
    dst->dtype = a->dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->shape, a->ndim)) return;

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

static void op_transpose(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return; 
    
    dst->dtype = a->dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->shape, a->ndim)) return;
    
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

static void op_inverse(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;
    
    dst->dtype = a->dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->shape, a->ndim)) return;

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
        // Fallback: Identity / Copy
        memcpy(dst->data, a->data, a->size * sizeof(f32));
    }
}

// Join(a, b) -> [..., 2] where ... is the common shape
static void op_join(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;

    size_t size = 1;
    if (a->size != b->size) {
        if (a->size != b->size) return; 
    }
    size = a->size;
    
    // Setup Output Shape
    for (int i=0; i<a->ndim; ++i) dst->shape[i] = a->shape[i];
    dst->shape[a->ndim] = 2;
    dst->ndim = a->ndim + 1;
    dst->size = size * 2;
    dst->dtype = a->dtype; 

    if (!mf_exec_ctx_resize_tensor(ctx, dst, dst->shape, dst->ndim)) return;
    
    f32* A = (f32*)a->data; 
    f32* B = (f32*)b->data; 
    f32* D = (f32*)dst->data;
    
    for (size_t i = 0; i < size; ++i) {
        D[i*2 + 0] = A[i];
        D[i*2 + 1] = B[i];
    }
}

void mf_ops_register_matrix(mf_op_func* table) {
    table[MF_OP_DOT] = op_dot;
    table[MF_OP_LENGTH] = op_length;
    table[MF_OP_MATMUL] = op_matmul; 
    table[MF_OP_TRANSPOSE] = op_transpose; 
    table[MF_OP_INVERSE] = op_inverse;
    table[MF_OP_JOIN] = op_join;
}
