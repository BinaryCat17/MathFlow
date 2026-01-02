#ifndef MF_KERNEL_UTILS_H
#define MF_KERNEL_UTILS_H

#include <mathflow/isa/mf_instruction.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <math.h>
#include <string.h>
#include <mathflow/isa/mf_exec_ctx.h>

// --- Stride Inference ---

#define MF_GET_STRIDE_D(inst)  ((inst)->strides[0])
#define MF_GET_STRIDE_S1(inst) ((inst)->strides[1])
#define MF_GET_STRIDE_S2(inst) ((inst)->strides[2])
#define MF_GET_STRIDE_S3(inst) ((inst)->strides[3])
#define MF_GET_STRIDE_S4(inst) ((inst)->strides[4])

// --- Macros: Optimized Kernel Definitions ---

#define MF_SAFE_F32(x) (isfinite(x) ? (x) : 0.0f)

#define MF_KERNEL_UNARY(NAME, TYPE_IN, TYPE_OUT, EXPR) \
void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    const size_t sz = ctx->batch_size; \
    TYPE_OUT* d_ptr = (TYPE_OUT*)ctx->reg_ptrs[inst->dest_idx]; \
    const TYPE_IN* a_ptr = (const TYPE_IN*)ctx->reg_ptrs[inst->src1_idx]; \
    const i32 st0 = MF_GET_STRIDE_D(inst); \
    const i32 st1 = MF_GET_STRIDE_S1(inst); \
    for(size_t i=0; i<sz; ++i) { \
        const TYPE_IN v = *a_ptr; \
        *d_ptr = (TYPE_OUT)(EXPR); \
        a_ptr += st1; d_ptr += st0; \
    } \
}

#define MF_KERNEL_BINARY(NAME, TYPE_IN, TYPE_OUT, EXPR) \
void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    const size_t sz = ctx->batch_size; \
    TYPE_OUT* d_ptr = (TYPE_OUT*)ctx->reg_ptrs[inst->dest_idx]; \
    const TYPE_IN* a_ptr = (const TYPE_IN*)ctx->reg_ptrs[inst->src1_idx]; \
    const TYPE_IN* b_ptr = (const TYPE_IN*)ctx->reg_ptrs[inst->src2_idx]; \
    const i32 st0 = MF_GET_STRIDE_D(inst); \
    const i32 st1 = MF_GET_STRIDE_S1(inst); \
    const i32 st2 = MF_GET_STRIDE_S2(inst); \
    for(size_t i=0; i<sz; ++i) { \
        const TYPE_IN va = *a_ptr; const TYPE_IN vb = *b_ptr; \
        *d_ptr = (TYPE_OUT)(EXPR); \
        a_ptr += st1; b_ptr += st2; d_ptr += st0; \
    } \
}

#define MF_KERNEL_TERNARY(NAME, TYPE_A, TYPE_B, TYPE_C, TYPE_OUT, EXPR) \
void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    const size_t sz = ctx->batch_size; \
    TYPE_OUT* d_ptr = (TYPE_OUT*)ctx->reg_ptrs[inst->dest_idx]; \
    const TYPE_A* a_ptr = (const TYPE_A*)ctx->reg_ptrs[inst->src1_idx]; \
    const TYPE_B* b_ptr = (const TYPE_B*)ctx->reg_ptrs[inst->src2_idx]; \
    const TYPE_C* c_ptr = (const TYPE_C*)ctx->reg_ptrs[inst->src3_idx]; \
    const i32 st0 = MF_GET_STRIDE_D(inst); \
    const i32 st1 = MF_GET_STRIDE_S1(inst); \
    const i32 st2 = MF_GET_STRIDE_S2(inst); \
    const i32 st3 = MF_GET_STRIDE_S3(inst); \
    for(size_t i=0; i<sz; ++i) { \
        const TYPE_A va = *a_ptr; const TYPE_B vb = *b_ptr; const TYPE_C vc = *c_ptr; \
        *d_ptr = (TYPE_OUT)(EXPR); \
        a_ptr += st1; b_ptr += st2; c_ptr += st3; d_ptr += st0; \
    } \
}

// --- Specific Shortcuts ---

#define MF_KERNEL_MATH_U(NAME, FUNC) MF_KERNEL_UNARY(NAME, f32, f32, MF_SAFE_F32(FUNC(v)))
#define MF_KERNEL_MATH_B(NAME, OP)   MF_KERNEL_BINARY(NAME, f32, f32, MF_SAFE_F32(va OP vb))
#define MF_KERNEL_MATH_BF(NAME, FUNC) MF_KERNEL_BINARY(NAME, f32, f32, MF_SAFE_F32(FUNC(va, vb)))

#define MF_KERNEL_COMPARE(NAME, OP) MF_KERNEL_BINARY(NAME, f32, u8, (va OP vb))
#define MF_KERNEL_LOGIC(NAME, OP)   MF_KERNEL_BINARY(NAME, u8, u8, (va OP vb))

#endif // MF_KERNEL_UTILS_H
