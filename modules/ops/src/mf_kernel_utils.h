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

#define MF_SAFE_F32(x) (isfinite((float)(x)) ? (f32)(x) : 0.0f)

#define MF_KERNEL_AUTO(NAME, EXPR, ARITY) \
void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    const size_t sz = ctx->batch_size; \
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx]; \
    const f32* a_ptr = (const f32*)ctx->reg_ptrs[inst->src1_idx]; \
    const f32* b_ptr = (ARITY >= 2) ? (const f32*)ctx->reg_ptrs[inst->src2_idx] : NULL; \
    const f32* c_ptr = (ARITY >= 3) ? (const f32*)ctx->reg_ptrs[inst->src3_idx] : NULL; \
    const i32 st0 = MF_GET_STRIDE_D(inst); \
    const i32 st1 = MF_GET_STRIDE_S1(inst); \
    const i32 st2 = (ARITY >= 2) ? MF_GET_STRIDE_S2(inst) : 0; \
    const i32 st3 = (ARITY >= 3) ? MF_GET_STRIDE_S3(inst) : 0; \
    for(size_t i=0; i<sz; ++i) { \
        const f32 va = *a_ptr; \
        const f32 vb = (ARITY >= 2) ? *b_ptr : 0.0f; \
        const f32 vc = (ARITY >= 3) ? *c_ptr : 0.0f; \
        *d_ptr = MF_SAFE_F32(EXPR); \
        a_ptr += st1; \
        if (ARITY >= 2) b_ptr += st2; \
        if (ARITY >= 3) c_ptr += st3; \
        d_ptr += st0; \
    } \
}

#endif // MF_KERNEL_UTILS_H