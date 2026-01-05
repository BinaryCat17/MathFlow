#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include "mf_ops_internal.h"
#include <math.h>

/**
 * MathFlow Atomic Kernels
 * Automatically generated from mf_ops_db.inc
 */

#define MF_GEN_AUTO(_op, _ke, _ar) MF_KERNEL_AUTO(_op, _ke, _ar)
#define MF_GEN_MANUAL(...)

#define MF_OP(_s, _n, _op, _cat, _strat, _in, _out, _tr, _sr, _ar, _p1, _p2, _p3, _p4, _kt, _ke, _arity) \
    MF_GEN_##_kt(_op, _ke, _arity)

MF_OP_LIST

#undef MF_OP
#undef MF_GEN_AUTO
#undef MF_GEN_MANUAL

// --- Vector Math (Custom Kernels) ---

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
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* a_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* b_ptr = (u8*)ctx->reg_ptrs[inst->src2_idx];
    
    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);
    i32 st2 = MF_GET_STRIDE_S2(inst);
    
    for (size_t i = 0; i < sz; ++i) {
        *(f32*)d_ptr = MF_SAFE_F32(_vec_dot_impl((f32*)a_ptr, (f32*)b_ptr, vec_len));
        a_ptr += st1;
        b_ptr += st2;
        d_ptr += st0;
    }
}

void op_LENGTH(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* a_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        *(f32*)d_ptr = MF_SAFE_F32(sqrtf(_vec_len_sq_impl((f32*)a_ptr, vec_len)));
        a_ptr += st1;
        d_ptr += st0;
    }
}

void op_NORMALIZE(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* a_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        f32 len = sqrtf(_vec_len_sq_impl((f32*)a_ptr, vec_len));
        f32 inv_len = (len > 1e-6f) ? (1.0f / len) : 0.0f;

        f32* d_f32 = (f32*)d_ptr;
        f32* a_f32 = (f32*)a_ptr;
        for (size_t j = 0; j < vec_len; ++j) {
            d_f32[j] = a_f32[j] * inv_len;
        }
        a_ptr += st1;
        d_ptr += st0;
    }
}

void op_SMOOTHSTEP(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* x_ptr = (u8*)ctx->reg_ptrs[inst->src2_idx];
    u8* e_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    const mf_type_info* e_info = &ctx->reg_info[inst->src1_idx];

    f32 e0 = 0.0f;
    f32 e1 = 1.0f;
    
    size_t e_count = 1;
    for(int i=0; i<e_info->ndim; ++i) e_count *= e_info->shape[i];

    if (e_count >= 2) {
        e0 = ((f32*)e_ptr)[0];
        e1 = ((f32*)e_ptr)[1];
    } else if (e_count == 1) {
        e1 = ((f32*)e_ptr)[0];
    }

    f32 span = e1 - e0;
    if (fabsf(span) < 1e-6f) span = 1e-6f;

    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st2 = MF_GET_STRIDE_S2(inst);

    for (size_t i = 0; i < sz; ++i) {
        f32 val = *(f32*)x_ptr;
        f32 t = (val - e0) / span;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        *(f32*)d_ptr = MF_SAFE_F32(t * t * (3.0f - 2.0f * t));
        
        x_ptr += st2;
        d_ptr += st0;
    }
}

// --- Reduction ---

void op_SUM(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* src_info = &ctx->reg_info[inst->src1_idx];
    size_t sz = ctx->batch_size;
    
    f32 sum = 0;
    u8* s_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    i32 st1 = MF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        sum += *(f32*)s_ptr;
        s_ptr += st1;
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr = sum;
}

void op_SIZE(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* src_info = &ctx->reg_info[inst->src1_idx];
    size_t count = 1;
    for (int i = 0; i < src_info->ndim; ++i) {
        count *= (src_info->shape[i] > 0 ? (size_t)src_info->shape[i] : 1);
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr = (f32)count;
}