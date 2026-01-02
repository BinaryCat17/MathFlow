#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>
#include <math.h>

typedef struct mf_tensor mf_tensor;
typedef struct mf_cpu_baked_instr mf_cpu_baked_instr;

static void op_matmul(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* a = bi->s1;
    mf_tensor* b = bi->s2;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    
    if (a->info.ndim != 2 || b->info.ndim != 2) {
        MF_LOG_ERROR("MatMul Error: Inputs must be 2D. Got %dD and %dD", a->info.ndim, b->info.ndim);
        ctx->error = MF_ERROR_SHAPE_MISMATCH;
        return;
    }

    int32_t M = a->info.shape[0];
    int32_t K = a->info.shape[1];
    int32_t N = b->info.shape[1];

    if (K != b->info.shape[0]) {
        MF_LOG_ERROR("MatMul Error: Shape mismatch. [%d,%d] x [%d,%d]", M, K, b->info.shape[0], N);
        ctx->error = MF_ERROR_SHAPE_MISMATCH;
        return;
    }

    int32_t out_shape[] = { M, N };
    dst->info.dtype = a->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, out_shape, 2)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    f32* base_a = (f32*)mf_tensor_data(a);
    f32* base_b = (f32*)mf_tensor_data(b);
    f32* base_c = (f32*)mf_tensor_data(dst);

    int32_t stride_ra = a->info.strides[0];
    int32_t stride_ka = a->info.strides[1];
    int32_t stride_kb = b->info.strides[0];
    int32_t stride_cb = b->info.strides[1];
    int32_t stride_rc = dst->info.strides[0];
    int32_t stride_cc = dst->info.strides[1];

    for (int32_t r = 0; r < M; r++) {
        for (int32_t c = 0; c < N; c++) { 
            float sum = 0.0f; 
            f32* pa = base_a + r * stride_ra;
            f32* pb = base_b + c * stride_cb;
            for (int32_t k = 0; k < K; k++) {
                sum += (*pa) * (*pb);
                pa += stride_ka;
                pb += stride_kb;
            }
            base_c[r * stride_rc + c * stride_cc] = sum; 
        }
    }
}

// Zero-Copy Transpose
static void op_transpose(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* a = bi->s1;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    
    // Use O(1) metadata swap
    if (!mf_tensor_transpose(dst, a)) {
        ctx->error = MF_ERROR_INVALID_OP;
    }
}

static void op_inverse(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* a = bi->s1;
    
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
        int32_t s0 = a->info.strides[0];
        int32_t s1 = a->info.strides[1];

        if (dim == 3) {
            mf_mat3 m;
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                m.m[r * 3 + c] = da[r * s0 + c * s1];
            }
            mf_mat3 res = mf_mat3_inverse(m);
            memcpy(mf_tensor_data(dst), res.m, sizeof(mf_mat3));
        } else {
            mf_mat4 m;
            for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) {
                m.m[r * 4 + c] = da[r * s0 + c * s1];
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

// Join(a, b, [c, d]) -> [..., N] where ... is the common shape
static void op_join(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* a = bi->s1;
    mf_tensor* b = bi->s2;
    mf_tensor* c = bi->s3;
    mf_tensor* d = bi->s4;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);

    int components = dst->info.shape[dst->info.ndim - 1];
    if (components >= 3) MF_CHECK_INPUT(ctx, c);
    if (components >= 4) MF_CHECK_INPUT(ctx, d);

    if (!mf_exec_ctx_resize_tensor(ctx, dst, dst->info.shape, dst->info.ndim)) return;
    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t sz_a = mf_tensor_count(a);
    mf_tensor_iter it_a = mf_tensor_iter_begin(a);
    mf_tensor_iter it_b = mf_tensor_iter_begin(b);
    mf_tensor_iter it_c = (components >= 3) ? mf_tensor_iter_begin(c) : (mf_tensor_iter){0};
    mf_tensor_iter it_d = (components >= 4) ? mf_tensor_iter_begin(d) : (mf_tensor_iter){0};
    mf_tensor_iter it_dst = mf_tensor_iter_begin(dst);
    
    for (size_t i = 0; i < sz_a; ++i) {
        *((f32*)it_dst.ptr) = *((f32*)it_a.ptr);
        mf_tensor_iter_next(&it_dst);
        
        *((f32*)it_dst.ptr) = *((f32*)it_b.ptr);
        mf_tensor_iter_next(&it_dst);

        if (components >= 3) {
            *((f32*)it_dst.ptr) = *((f32*)it_c.ptr);
            mf_tensor_iter_next(&it_dst);
            mf_tensor_iter_next(&it_c);
        }
        
        if (components >= 4) {
            *((f32*)it_dst.ptr) = *((f32*)it_d.ptr);
            mf_tensor_iter_next(&it_dst);
            mf_tensor_iter_next(&it_d);
        }

        mf_tensor_iter_next(&it_a);
        mf_tensor_iter_next(&it_b);
    }
}

void mf_ops_register_matrix(mf_op_func* table) {
    table[MF_OP_MATMUL] = op_matmul; 
    table[MF_OP_TRANSPOSE] = op_transpose; 
    table[MF_OP_INVERSE] = op_inverse;
    table[MF_OP_JOIN] = op_join;
}
