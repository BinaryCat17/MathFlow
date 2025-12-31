#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>
#include <math.h>

// Helper to densify tensor if needed (Fallback for Ops that don't support strides yet)
static bool ensure_contiguous(mf_exec_ctx* ctx, const mf_tensor* src, mf_tensor* temp_out) {
    // Check if contiguous
    size_t count = mf_tensor_count(src);
    // Simple check: Last stride is 1 and others match shape? 
    // Or just check if we can iterate linearly.
    // For now, ALWAYS copy to temp buffer if strides != standard row-major
    
    // Check standard strides
    int32_t stride = 1;
    bool standard = true;
    for (int k = src->info.ndim - 1; k >= 0; --k) {
        if (src->info.strides[k] != stride) { standard = false; break; }
        stride *= src->info.shape[k];
    }
    
    if (standard) {
        *temp_out = *src; // View
        return true;
    }
    
    // Alloc temp contiguous
    if (!mf_tensor_alloc(temp_out, ctx->allocator, &src->info)) return false;
    
    // Copy data (slow path)
    // We need an iterator that respects src strides... 
    // But since we don't have one readily available for N-dims, 
    // we limit Transpose/Strides to 2D for now in the Ops support.
    
    // 2D Copy
    if (src->info.ndim == 2) {
        f32* dst_ptr = (f32*)mf_tensor_data(temp_out);
        f32* src_base = (f32*)mf_tensor_data(src);
        int rows = src->info.shape[0];
        int cols = src->info.shape[1];
        int s0 = src->info.strides[0];
        int s1 = src->info.strides[1];
        
        for(int r=0; r<rows; ++r) {
            for(int c=0; c<cols; ++c) {
                dst_ptr[r*cols + c] = src_base[r*s0 + c*s1];
            }
        }
        return true;
    }
    
    return false; // Unsupported rank for strided access
}

// Dot(a, b) -> Sum(a*b) along last axis
static void op_dot(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    
    size_t sz_a = mf_tensor_count(a);
    size_t sz_b = mf_tensor_count(b);
    if (sz_a != sz_b) return; 

    // Handle Strides (Densify inputs)
    mf_tensor a_cont, b_cont;
    if (!ensure_contiguous(ctx, a, &a_cont)) return;
    if (!ensure_contiguous(ctx, b, &b_cont)) return;

    int out_ndim = (a->info.ndim > 0) ? a->info.ndim - 1 : 0;
    
    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, (uint8_t)out_ndim)) return;

    f32* A = (f32*)mf_tensor_data(&a_cont); 
    f32* B = (f32*)mf_tensor_data(&b_cont); 
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
static void op_length(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;

    mf_tensor a_cont;
    if (!ensure_contiguous(ctx, a, &a_cont)) return;

    int out_ndim = (a->info.ndim > 0) ? a->info.ndim - 1 : 0;
    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, (uint8_t)out_ndim)) return;

    f32* A = (f32*)mf_tensor_data(&a_cont); 
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

static void op_matmul(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    
    size_t sz_a = mf_tensor_count(a);
    int dim = (int)sqrtf((float)sz_a); 
    if (dim * dim != (int)sz_a) return; 
    
    mf_tensor a_cont, b_cont;
    if (!ensure_contiguous(ctx, a, &a_cont)) return;
    if (!ensure_contiguous(ctx, b, &b_cont)) return;

    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim)) return;

    // Fast Path
    if (dim == 4 && sz_a == 16) {
        mf_mat4 A, B; 
        memcpy(A.m, mf_tensor_data(&a_cont), sizeof(mf_mat4));
        memcpy(B.m, mf_tensor_data(&b_cont), sizeof(mf_mat4));
        mf_mat4 R = mf_mat4_mul(A, B);
        memcpy(mf_tensor_data(dst), R.m, sizeof(mf_mat4));
        return;
    }
    if (dim == 3 && sz_a == 9) {
        mf_mat3 A, B; 
        memcpy(A.m, mf_tensor_data(&a_cont), sizeof(mf_mat3));
        memcpy(B.m, mf_tensor_data(&b_cont), sizeof(mf_mat3));
        mf_mat3 R = mf_mat3_mul(A, B);
        memcpy(mf_tensor_data(dst), R.m, sizeof(mf_mat3));
        return;
    }

    // Generic Path
    f32* A = (f32*)mf_tensor_data(&a_cont); 
    f32* B = (f32*)mf_tensor_data(&b_cont); 
    f32* C = (f32*)mf_tensor_data(dst);
    for (int r = 0; r < dim; r++) for (int c = 0; c < dim; c++) { 
        float sum = 0.0f; 
        for (int k = 0; k < dim; k++) sum += A[r * dim + k] * B[k * dim + c]; 
        C[r * dim + c] = sum; 
    }
}

// Zero-Copy Transpose
static void op_transpose(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return; 
    
    // Use O(1) metadata swap
    // Ops consuming this must support strides (via ensure_contiguous fallback)
    if (!mf_tensor_transpose(dst, a)) {
        // Fallback or error?
        // If 1D or >2D, mf_tensor_transpose might fail currently (implementation limited to 2D)
        ctx->error = MF_ERROR_INVALID_OP;
    }
}

static void op_inverse(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return;
    
    mf_tensor a_cont;
    if (!ensure_contiguous(ctx, a, &a_cont)) return;

    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim)) return;

    size_t sz_a = mf_tensor_count(a);
    int dim = (int)sqrtf((float)sz_a);
    
    if (dim == 3 && sz_a == 9) {
        mf_mat3 m;
        memcpy(m.m, mf_tensor_data(&a_cont), sizeof(mf_mat3));
        mf_mat3 res = mf_mat3_inverse(m);
        memcpy(mf_tensor_data(dst), res.m, sizeof(mf_mat3));
    } 
    else if (dim == 4 && sz_a == 16) {
        mf_mat4 m;
        memcpy(m.m, mf_tensor_data(&a_cont), sizeof(mf_mat4));
        mf_mat4 res = mf_mat4_inverse(m);
        memcpy(mf_tensor_data(dst), res.m, sizeof(mf_mat4));
    }
    else {
        // Fallback: Identity / Copy
        memcpy(mf_tensor_data(dst), mf_tensor_data(&a_cont), sz_a * sizeof(f32));
    }
}

// Join(a, b) -> [..., 2] where ... is the common shape
static void op_join(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;

    size_t sz_a = mf_tensor_count(a);
    size_t sz_b = mf_tensor_count(b);
    if (sz_a != sz_b) return; 
    
    // Densify? Join requires copy anyway, so iterating strided source is better than 
    // densify->copy. But for now, reuse ensure_contiguous for safety.
    mf_tensor a_cont, b_cont;
    if (!ensure_contiguous(ctx, a, &a_cont)) return;
    if (!ensure_contiguous(ctx, b, &b_cont)) return;
    
    // Setup Output Shape
    int32_t out_shape[MF_MAX_DIMS];
    for (int i=0; i<a->info.ndim; ++i) out_shape[i] = a->info.shape[i];
    out_shape[a->info.ndim] = 2;
    uint8_t out_ndim = a->info.ndim + 1;

    dst->info.dtype = a->info.dtype; 
    if (!mf_exec_ctx_resize_tensor(ctx, dst, out_shape, out_ndim)) return;
    
    f32* A = (f32*)mf_tensor_data(&a_cont); 
    f32* B = (f32*)mf_tensor_data(&b_cont); 
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