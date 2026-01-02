#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>
#include <math.h>

static void op_matmul(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    const mf_type_info* b_info = &ctx->reg_info[inst->src2_idx];
    const mf_type_info* dst_info = &ctx->reg_info[inst->dest_idx];

    const int32_t M = a_info->shape[a_info->ndim - 2];
    const int32_t K = a_info->shape[a_info->ndim - 1];
    const int32_t N = b_info->shape[b_info->ndim - 1];

    f32* base_a = (f32*)ctx->reg_ptrs[inst->src1_idx];
    f32* base_b = (f32*)ctx->reg_ptrs[inst->src2_idx];
    f32* base_c = (f32*)ctx->reg_ptrs[inst->dest_idx];

    const i32 st_batch_a = MF_GET_STRIDE_S1(inst);
    const i32 st_batch_b = MF_GET_STRIDE_S2(inst);
    const i32 st_batch_c = MF_GET_STRIDE_D(inst);

    const int32_t stride_ra = a_info->strides[a_info->ndim - 2];
    const int32_t stride_ka = a_info->strides[a_info->ndim - 1];
    const int32_t stride_kb = b_info->strides[b_info->ndim - 2];
    const int32_t stride_cb = b_info->strides[b_info->ndim - 1];
    const int32_t stride_rc = dst_info->strides[dst_info->ndim - 2];
    const int32_t stride_cc = dst_info->strides[dst_info->ndim - 1];

    const size_t batch_size = ctx->batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
        f32* curr_a = base_a + b * st_batch_a;
        f32* curr_b = base_b + b * st_batch_b;
        f32* curr_c = base_c + b * st_batch_c;

        for (int32_t r = 0; r < M; r++) {
            for (int32_t c = 0; c < N; c++) { 
                float sum = 0.0f; 
                f32* pa = curr_a + r * stride_ra;
                f32* pb = curr_b + c * stride_cb;
                for (int32_t k = 0; k < K; k++) {
                    sum += (*pa) * (*pb);
                    pa += stride_ka;
                    pb += stride_kb;
                }
                curr_c[r * stride_rc + c * stride_cc] = sum; 
            }
        }
    }
}

// Zero-Copy Transpose
static void op_transpose(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    // Transpose is a metadata operation. In our new flat model, 
    // it should be handled by the dispatcher/compiler during stride calculation.
    // If it's a kernel, it's essentially a no-op if strides are already correct.
    (void)ctx; (void)inst;
}

static void op_inverse(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t sz_a = 1;
    for(int i=0; i<a_info->ndim; ++i) sz_a *= a_info->shape[i];

    int dim = (int)sqrtf((float)sz_a);
    f32* da = (f32*)ctx->reg_ptrs[inst->src1_idx];
    f32* dd = (f32*)ctx->reg_ptrs[inst->dest_idx];

    if ((dim == 3 && sz_a == 9) || (dim == 4 && sz_a == 16)) {
        int32_t s0 = a_info->strides[0];
        int32_t s1 = a_info->strides[1];

        if (dim == 3) {
            mf_mat3 m;
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                m.m[r * 3 + c] = da[r * s0 + c * s1];
            }
            mf_mat3 res = mf_mat3_inverse(m);
            memcpy(dd, res.m, sizeof(mf_mat3));
        } else {
            mf_mat4 m;
            for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) {
                m.m[r * 4 + c] = da[r * s0 + c * s1];
            }
            mf_mat4 res = mf_mat4_inverse(m);
            memcpy(dd, res.m, sizeof(mf_mat4));
        }
    }
    else {
        // Fallback: Copy with strides (Flattening)
        size_t count = sz_a;
        int32_t indices[MF_MAX_DIMS] = {0};
        for(size_t i=0; i<count; ++i) {
            size_t offset = 0;
            size_t temp_idx = i;
            for (int d = a_info->ndim - 1; d >= 0; --d) {
                offset += (temp_idx % a_info->shape[d]) * a_info->strides[d];
                temp_idx /= a_info->shape[d];
            }
            dd[i] = da[offset];
        }
    }
}

// Join(a, b, [c, d]) -> [..., N] where ... is the common shape
static void op_join(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* dst_info = &ctx->reg_info[inst->dest_idx];
    int components = dst_info->shape[dst_info->ndim - 1];
    
    size_t sz_a = ctx->batch_size;
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    f32* a_ptr = (f32*)ctx->reg_ptrs[inst->src1_idx];
    f32* b_ptr = (f32*)ctx->reg_ptrs[inst->src2_idx];
    f32* c_ptr = (components >= 3) ? (f32*)ctx->reg_ptrs[inst->src3_idx] : NULL;
    f32* d_in_ptr = (components >= 4) ? (f32*)ctx->reg_ptrs[inst->src4_idx] : NULL;

    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);
    i32 st2 = MF_GET_STRIDE_S2(inst);
    i32 st3 = MF_GET_STRIDE_S3(inst);
    i32 st4 = MF_GET_STRIDE_S4(inst); // Now available in ISA v4!

    for (size_t i = 0; i < sz_a; ++i) {
        d_ptr[0] = *a_ptr;
        d_ptr[1] = *b_ptr;
        if (components >= 3) d_ptr[2] = *c_ptr;
        if (components >= 4) d_ptr[3] = *d_in_ptr;

        a_ptr += st1;
        b_ptr += st2;
        if (c_ptr) c_ptr += st3;
        if (d_in_ptr) d_in_ptr += st4;
        d_ptr += st0;
    }
}

void mf_ops_register_matrix(mf_op_func* table) {
    table[MF_OP_MATMUL] = op_matmul; 
    table[MF_OP_TRANSPOSE] = op_transpose; 
    table[MF_OP_INVERSE] = op_inverse;
    table[MF_OP_JOIN] = op_join;
}