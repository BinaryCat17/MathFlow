#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>
#include <math.h>

// Dot(a, b) -> Sum(a*b) along last axis
static void op_dot(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    
    size_t sz_a = mf_tensor_count(a);
    size_t sz_b = mf_tensor_count(b);
    if (sz_a != sz_b) return; 

    int out_ndim = (a->info.ndim > 0) ? a->info.ndim - 1 : 0;
    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, (uint8_t)out_ndim)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    f32* D = (f32*)mf_tensor_data(dst);
    size_t dim = (a->info.ndim <= 1) ? sz_a : (size_t)a->info.shape[a->info.ndim-1];
    size_t batch = sz_a / dim;
    
    mf_tensor_iter it_a = mf_tensor_iter_begin(a);
    mf_tensor_iter it_b = mf_tensor_iter_begin(b);

    for (size_t i = 0; i < batch; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            sum += (*((f32*)it_a.ptr)) * (*((f32*)it_b.ptr));
            mf_tensor_iter_next(&it_a);
            mf_tensor_iter_next(&it_b);
        }
        D[i] = sum;
    }
}

// Length(a) -> Sqrt(Dot(a, a))
static void op_length(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);

    int out_ndim = (a->info.ndim > 0) ? a->info.ndim - 1 : 0;
    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, (uint8_t)out_ndim)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    f32* D = (f32*)mf_tensor_data(dst);
    size_t sz_a = mf_tensor_count(a);
    size_t dim = (a->info.ndim <= 1) ? sz_a : (size_t)a->info.shape[a->info.ndim-1];
    size_t batch = sz_a / dim;
    
    mf_tensor_iter it_a = mf_tensor_iter_begin(a);

    for (size_t i = 0; i < batch; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            float val = *((f32*)it_a.ptr);
            sum += val * val;
            mf_tensor_iter_next(&it_a);
        }
        D[i] = sqrtf(sum);
    }
}

static void op_matmul(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    
    if (a->info.ndim != 2 || b->info.ndim != 2) return;

    int32_t M = a->info.shape[0];
    int32_t K = a->info.shape[1];
    int32_t N = b->info.shape[1];

    if (K != b->info.shape[0]) return;

    int32_t out_shape[] = { M, N };
    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, out_shape, 2)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    f32* C = (f32*)mf_tensor_data(dst);
    f32* da = (f32*)mf_tensor_data(a);
    f32* db = (f32*)mf_tensor_data(b);

    for (int32_t r = 0; r < M; r++) {
        for (int32_t c = 0; c < N; c++) { 
            float sum = 0.0f; 
            for (int32_t k = 0; k < K; k++) {
                int32_t idx_a[2] = {r, k};
                int32_t idx_b[2] = {k, c};
                sum += da[mf_tensor_get_offset(a, idx_a)] * db[mf_tensor_get_offset(b, idx_b)];
            }
            C[r * N + c] = sum; 
        }
    }
}

// Zero-Copy Transpose
static void op_transpose(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    
    // Use O(1) metadata swap
    if (!mf_tensor_transpose(dst, a)) {
        ctx->error = MF_ERROR_INVALID_OP;
    }
}

static void op_inverse(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    
    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    size_t sz_a = mf_tensor_count(a);
    int dim = (int)sqrtf((float)sz_a);
    f32* da = (f32*)mf_tensor_data(a);

    if ((dim == 3 && sz_a == 9) || (dim == 4 && sz_a == 16)) {
        // Densify to local mat for inverse call
        if (dim == 3) {
            mf_mat3 m;
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                int32_t idx[2] = {r, c};
                m.m[r * 3 + c] = da[mf_tensor_get_offset(a, idx)];
            }
            mf_mat3 res = mf_mat3_inverse(m);
            memcpy(mf_tensor_data(dst), res.m, sizeof(mf_mat3));
        } else {
            mf_mat4 m;
            for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) {
                int32_t idx[2] = {r, c};
                m.m[r * 4 + c] = da[mf_tensor_get_offset(a, idx)];
            }
            mf_mat4 res = mf_mat4_inverse(m);
            memcpy(mf_tensor_data(dst), res.m, sizeof(mf_mat4));
        }
    }
    else {
        // Fallback: Copy with strides
        f32* dd = (f32*)mf_tensor_data(dst);
        mf_tensor_iter it_a = mf_tensor_iter_begin(a);
        for(size_t i=0; i<sz_a; ++i) {
            dd[i] = *((f32*)it_a.ptr);
            mf_tensor_iter_next(&it_a);
        }
    }
}

// Join(a, b) -> [..., 2] where ... is the common shape
static void op_join(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);

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
    MF_CHECK_DST_DATA(ctx, dst);
    
    f32* D = (f32*)mf_tensor_data(dst);
    mf_tensor_iter it_a = mf_tensor_iter_begin(a);
    mf_tensor_iter it_b = mf_tensor_iter_begin(b);
    
    for (size_t i = 0; i < sz_a; ++i) {
        D[i*2 + 0] = *((f32*)it_a.ptr);
        D[i*2 + 1] = *((f32*)it_b.ptr);
        mf_tensor_iter_next(&it_a);
        mf_tensor_iter_next(&it_b);
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