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
    
    size_t sz_a = mf_tensor_count(a);
    size_t sz_b = mf_tensor_count(b);
    if (sz_a != sz_b) return; 

    int out_ndim = (a->info.ndim > 0) ? a->info.ndim - 1 : 0;
    
    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, (uint8_t)out_ndim)) return;

    f32* A = (f32*)mf_tensor_data(a); 
    f32* B = (f32*)mf_tensor_data(b); 
    f32* D = (f32*)mf_tensor_data(dst);
    
    size_t dim = (a->info.ndim <= 1) ? sz_a : (size_t)a->info.shape[a->info.ndim-1];
    size_t batch = sz_a / dim;
    
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

    int out_ndim = (a->info.ndim > 0) ? a->info.ndim - 1 : 0;
    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, (uint8_t)out_ndim)) return;

    f32* A = (f32*)mf_tensor_data(a); 
    f32* D = (f32*)mf_tensor_data(dst);
    
    size_t sz_a = mf_tensor_count(a);
    size_t dim = (a->info.ndim <= 1) ? sz_a : (size_t)a->info.shape[a->info.ndim-1];
    size_t batch = sz_a / dim;
    
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
    
    size_t sz_a = mf_tensor_count(a);
    int dim = (int)sqrtf((float)sz_a); 
    if (dim * dim != (int)sz_a) return; 
    
    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim)) return;

    // Fast Path
    if (dim == 4 && sz_a == 16) {
        mf_mat4 A, B; 
        memcpy(A.m, mf_tensor_data(a), sizeof(mf_mat4));
        memcpy(B.m, mf_tensor_data(b), sizeof(mf_mat4));
        mf_mat4 R = mf_mat4_mul(A, B);
        memcpy(mf_tensor_data(dst), R.m, sizeof(mf_mat4));
        return;
    }
    if (dim == 3 && sz_a == 9) {
        mf_mat3 A, B; 
        memcpy(A.m, mf_tensor_data(a), sizeof(mf_mat3));
        memcpy(B.m, mf_tensor_data(b), sizeof(mf_mat3));
        mf_mat3 R = mf_mat3_mul(A, B);
        memcpy(mf_tensor_data(dst), R.m, sizeof(mf_mat3));
        return;
    }

    // Generic Path
    f32* A = (f32*)mf_tensor_data(a); 
    f32* B = (f32*)mf_tensor_data(b); 
    f32* C = (f32*)mf_tensor_data(dst);
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
    
    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim)) return;
    
    size_t sz_a = mf_tensor_count(a);
    int dim = (int)sqrtf((float)sz_a);
    
    // Fast Path
    if (dim == 4 && sz_a == 16) {
        mf_mat4 A; 
        memcpy(A.m, mf_tensor_data(a), sizeof(mf_mat4));
        mf_mat4 R = mf_mat4_transpose(A);
        memcpy(mf_tensor_data(dst), R.m, sizeof(mf_mat4));
        return;
    }
    if (dim == 3 && sz_a == 9) {
        mf_mat3 A; 
        memcpy(A.m, mf_tensor_data(a), sizeof(mf_mat3));
        mf_mat3 R = mf_mat3_transpose(A);
        memcpy(mf_tensor_data(dst), R.m, sizeof(mf_mat3));
        return;
    }

    // Generic Path
    f32* src = (f32*)mf_tensor_data(a); 
    f32* out = (f32*)mf_tensor_data(dst);
    for (int r = 0; r < dim; r++) for (int c = 0; c < dim; c++) out[c * dim + r] = src[r * dim + c];
}

static void op_inverse(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;
    
    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim)) return;

    size_t sz_a = mf_tensor_count(a);
    int dim = (int)sqrtf((float)sz_a);
    
    if (dim == 3 && sz_a == 9) {
        mf_mat3 m;
        memcpy(m.m, mf_tensor_data(a), sizeof(mf_mat3));
        mf_mat3 res = mf_mat3_inverse(m);
        memcpy(mf_tensor_data(dst), res.m, sizeof(mf_mat3));
    } 
    else if (dim == 4 && sz_a == 16) {
        mf_mat4 m;
        memcpy(m.m, mf_tensor_data(a), sizeof(mf_mat4));
        mf_mat4 res = mf_mat4_inverse(m);
        memcpy(mf_tensor_data(dst), res.m, sizeof(mf_mat4));
    }
    else {
        // Fallback: Identity / Copy
        memcpy(mf_tensor_data(dst), mf_tensor_data(a), sz_a * sizeof(f32));
    }
}

// Join(a, b) -> [..., 2] where ... is the common shape
static void op_join(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;

    size_t sz_a = mf_tensor_count(a);
    size_t sz_b = mf_tensor_count(b);
    if (sz_a != sz_b) return; 
    
    // Setup Output Shape
    int32_t out_shape[MF_MAX_DIMS];
    for (int i=0; i<a->info.ndim; ++i) out_shape[i] = a->info.shape[i];
    out_shape[a->info.ndim] = 2;
    uint8_t out_ndim = a->info.ndim + 1;

    dst->info.dtype = a->info.dtype; 
    if (!mf_exec_ctx_resize_tensor(ctx, dst, out_shape, out_ndim)) return;
    
    f32* A = (f32*)mf_tensor_data(a); 
    f32* B = (f32*)mf_tensor_data(b); 
    f32* D = (f32*)mf_tensor_data(dst);
    
    for (size_t i = 0; i < sz_a; ++i) {
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