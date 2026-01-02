#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include "mf_ops_internal.h"
#include <math.h>

// --- Arithmetic ---
MF_KERNEL_MATH_B(ADD, +)
MF_KERNEL_MATH_B(SUB, -)
MF_KERNEL_MATH_B(MUL, *)
MF_KERNEL_MATH_B(DIV, /)
MF_KERNEL_BINARY(STEP, f32, f32, MF_SAFE_F32(vb < va ? 0.0f : 1.0f))
MF_KERNEL_MATH_BF(ATAN2, atan2f)
MF_KERNEL_MATH_BF(POW, powf)

// --- Unary Math ---
MF_KERNEL_MATH_U(SIN, sinf)
MF_KERNEL_MATH_U(COS, cosf)
MF_KERNEL_MATH_U(FLOOR, floorf)
MF_KERNEL_MATH_U(CEIL, ceilf)
MF_KERNEL_MATH_U(ABS, fabsf)
MF_KERNEL_MATH_U(SQRT, sqrtf)

// --- Min/Max/Clamp/Mix ---
MF_KERNEL_BINARY(MIN, f32, f32, (va < vb ? va : vb))
MF_KERNEL_BINARY(MAX, f32, f32, (va > vb ? va : vb))
MF_KERNEL_TERNARY(FMA, f32, f32, f32, f32, MF_SAFE_F32(fmaf(va, vb, vc)))
MF_KERNEL_TERNARY(CLAMP, f32, f32, f32, f32, MF_SAFE_F32(fminf(fmaxf(va, vb), vc)))
MF_KERNEL_TERNARY(MIX, f32, f32, f32, f32, MF_SAFE_F32(va * (1.0f - vc) + vb * vc))

// --- Vector Math ---

static inline f32 _vec_dot_impl(f32* a_ptr, f32* b_ptr, size_t len) {
    f32 sum = 0;
    for (size_t j = 0; j < len; ++j) {
        sum += a_ptr[j] * b_ptr[j];
    }
    return sum;
}

static inline f32 _vec_len_sq_impl(f32* a_ptr, size_t len) {
    f32 sum = 0;
    for (size_t j = 0; j < len; ++j) {
        f32 v = a_ptr[j];
        sum += v * v;
    }
    return sum;
}

void op_DOT(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    f32* a_ptr = (f32*)ctx->reg_ptrs[inst->src1_idx];
    f32* b_ptr = (f32*)ctx->reg_ptrs[inst->src2_idx];
    
    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);
    i32 st2 = MF_GET_STRIDE_S2(inst);
    
    for (size_t i = 0; i < sz; ++i) {
        *d_ptr = MF_SAFE_F32(_vec_dot_impl(a_ptr, b_ptr, vec_len));
        a_ptr += st1;
        b_ptr += st2;
        d_ptr += st0;
    }
}

void op_LENGTH(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    f32* a_ptr = (f32*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        *d_ptr = MF_SAFE_F32(sqrtf(_vec_len_sq_impl(a_ptr, vec_len)));
        a_ptr += st1;
        d_ptr += st0;
    }
}

void op_NORMALIZE(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    f32* a_ptr = (f32*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        f32 len = sqrtf(_vec_len_sq_impl(a_ptr, vec_len));
        f32 inv_len = (len > 1e-6f) ? (1.0f / len) : 0.0f;

        for (size_t j = 0; j < vec_len; ++j) {
            d_ptr[j] = a_ptr[j] * inv_len;
        }
        a_ptr += st1;
        d_ptr += st0;
    }
}

void op_SMOOTHSTEP(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    size_t sz = ctx->batch_size;
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    f32* x_ptr = (f32*)ctx->reg_ptrs[inst->src2_idx];
    f32* e_ptr = (f32*)ctx->reg_ptrs[inst->src1_idx];
    const mf_type_info* e_info = &ctx->reg_info[inst->src1_idx];

    f32 e0 = 0.0f;
    f32 e1 = 1.0f;
    
    size_t e_count = 1;
    for(int i=0; i<e_info->ndim; ++i) e_count *= e_info->shape[i];

    if (e_count >= 2) {
        e0 = e_ptr[0];
        e1 = e_ptr[1];
    } else if (e_count == 1) {
        e1 = e_ptr[0];
    }

    f32 span = e1 - e0;
    if (fabsf(span) < 1e-6f) span = 1e-6f;

    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st2 = MF_GET_STRIDE_S2(inst);

    for (size_t i = 0; i < sz; ++i) {
        f32 val = *x_ptr;
        f32 t = (val - e0) / span;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        *d_ptr = MF_SAFE_F32(t * t * (3.0f - 2.0f * t));
        
        x_ptr += st2;
        d_ptr += st0;
    }
}

// --- Reduction ---

void op_SUM(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* src_info = &ctx->reg_info[inst->src1_idx];
    size_t sz = ctx->batch_size;
    
    f32 sum = 0;
    f32* s_ptr = (f32*)ctx->reg_ptrs[inst->src1_idx];
    i32 st1 = MF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        sum += *s_ptr;
        s_ptr += st1;
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr = sum;
}
