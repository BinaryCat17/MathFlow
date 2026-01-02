#ifndef MF_KERNEL_UTILS_H
#define MF_KERNEL_UTILS_H

#include <mathflow/isa/mf_instruction.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <math.h>
#include <mathflow/isa/mf_accessor.h>
#include <string.h>
#include <mathflow/isa/mf_exec_ctx.h>

// --- Helper: Shape Resolution (Inline) ---

static inline bool mf_utils_resolve_binary_shape(mf_exec_ctx* ctx, mf_tensor* dst, const mf_tensor* a, const mf_tensor* b) {
    if (dst->info.dtype == MF_DTYPE_UNKNOWN) dst->info.dtype = a->info.dtype;

    size_t count_a = mf_tensor_count(a);
    size_t count_b = mf_tensor_count(b);
    const mf_tensor* shape_src = (count_a >= count_b) ? a : b;
    return mf_exec_ctx_resize_tensor(ctx, dst, shape_src->info.shape, shape_src->info.ndim);
}

static inline bool mf_utils_resolve_ternary_shape(mf_exec_ctx* ctx, mf_tensor* dst, const mf_tensor* a, const mf_tensor* b, const mf_tensor* c) {
    if (dst->info.dtype == MF_DTYPE_UNKNOWN) dst->info.dtype = b->info.dtype;

    size_t count_a = mf_tensor_count(a);
    size_t count_b = mf_tensor_count(b);
    size_t count_c = mf_tensor_count(c);
    
    size_t max_count = count_a;
    const mf_tensor* shape_src = a;
    
    if (count_b > max_count) { max_count = count_b; shape_src = b; }
    if (count_c > max_count) { max_count = count_c; shape_src = c; }
    
    return mf_exec_ctx_resize_tensor(ctx, dst, shape_src->info.shape, shape_src->info.ndim);
}

static inline bool mf_utils_resolve_unary_shape(mf_exec_ctx* ctx, mf_tensor* dst, const mf_tensor* a) {
    if (dst->info.dtype == MF_DTYPE_UNKNOWN) dst->info.dtype = a->info.dtype;
    return mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim);
}

// --- Helper: Scalar Extraction ---

static inline int32_t mf_utils_get_scalar_int(const mf_tensor* t) {
    if (!t || !t->buffer || !t->buffer->data) return 0;
    void* d = mf_tensor_data(t);
    if (t->info.dtype == MF_DTYPE_I32) return *((int32_t*)d);
    if (t->info.dtype == MF_DTYPE_F32) return (int32_t)(*((f32*)d));
    if (t->info.dtype == MF_DTYPE_U8) return (int32_t)(*((u8*)d));
    return 0;
}

static inline f32 mf_utils_get_scalar_f32(const mf_tensor* t) {
    if (!t || !t->buffer || !t->buffer->data) return 0.0f;
    void* d = mf_tensor_data(t);
    if (t->info.dtype == MF_DTYPE_F32) return *((f32*)d);
    if (t->info.dtype == MF_DTYPE_I32) return (f32)(*((int32_t*)d));
    if (t->info.dtype == MF_DTYPE_U8) return (f32)(*((u8*)d));
    return 0.0f;
}

// --- Stride Inference ---

#define MF_GET_STRIDE_D(inst)  ((inst)->strides[0])
#define MF_GET_STRIDE_S1(inst) ((inst)->strides[1])
#define MF_GET_STRIDE_S2(inst) ((inst)->strides[2])
#define MF_GET_STRIDE_S3(inst) ((inst)->strides[3])

// --- Macros: Optimized Kernel Definitions ---

#define MF_SAFE_F32(x) (isfinite(x) ? (x) : 0.0f)

#define MF_CHECK_COMMON(ctx, dst, s1) \
    MF_CHECK_DST_VIEW(ctx, dst); \
    MF_CHECK_INPUT(ctx, s1); \
    dst->info.dtype = MF_DTYPE_F32;

#define MF_RESOLVE_VEC_SHAPE(ctx, dst, src, ndim_offset) \
    if ((src)->info.ndim > 0) { \
        if (!mf_exec_ctx_resize_tensor(ctx, dst, (src)->info.shape, (src)->info.ndim + (ndim_offset))) return; \
    } else { \
        u32 scalar_shape[] = {1}; \
        if (!mf_exec_ctx_resize_tensor(ctx, dst, scalar_shape, 1)) return; \
    } \
    MF_CHECK_DST_DATA(ctx, dst);

#define MF_KERNEL_VECTOR_REDUCE(NAME, EXPR) \
static void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    mf_tensor* dst = &ctx->registers[inst->dest_idx]; \
    mf_tensor* a = &ctx->registers[inst->src1_idx]; \
    mf_tensor* b = (inst->src2_idx > 0) ? &ctx->registers[inst->src2_idx] : NULL; \
    MF_CHECK_COMMON(ctx, dst, a); \
    if (b) MF_CHECK_INPUT(ctx, b); \
    MF_RESOLVE_VEC_SHAPE(ctx, dst, a, -1); \
    size_t vec_len = a->info.shape[a->info.ndim - 1]; \
    size_t out_count = mf_tensor_count(dst); \
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst); \
    mf_accessor_f32 it_a = mf_accessor_f32_begin(a); \
    mf_accessor_f32 it_b = b ? mf_accessor_f32_begin(b) : it_a; \
    i32 st0 = MF_GET_STRIDE_D(inst); \
    for (size_t i = 0; i < out_count; ++i) { \
        mf_accessor_f32_set(&it_dst, MF_SAFE_F32(EXPR)); \
        mf_accessor_f32_advance(&it_dst, st0); \
    } \
}

#define MF_KERNEL_BINARY_GENERIC(NAME, TYPE_IN, TYPE_OUT, DTYPE_OUT, EXPR, ACC_IN, ACC_OUT) \
static void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    mf_tensor* dst = &ctx->registers[inst->dest_idx]; \
    mf_tensor* a = &ctx->registers[inst->src1_idx]; \
    mf_tensor* b = &ctx->registers[inst->src2_idx]; \
    MF_CHECK_DST_VIEW(ctx, dst); \
    MF_CHECK_INPUT(ctx, a); \
    MF_CHECK_INPUT(ctx, b); \
    dst->info.dtype = MF_DTYPE_##DTYPE_OUT; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    i32 st0 = MF_GET_STRIDE_D(inst); \
    i32 st1 = MF_GET_STRIDE_S1(inst); \
    i32 st2 = MF_GET_STRIDE_S2(inst); \
    if (st0 == 1 && st1 == 1 && st2 == 1 && dst->byte_offset == 0 && a->byte_offset == 0 && b->byte_offset == 0) { \
        TYPE_OUT* d_ptr = (TYPE_OUT*)dst->buffer->data; \
        TYPE_IN* a_ptr = (TYPE_IN*)a->buffer->data; \
        TYPE_IN* b_ptr = (TYPE_IN*)b->buffer->data; \
        for(size_t i=0; i<sz_dst; ++i) { \
            TYPE_IN va = a_ptr[i]; TYPE_IN vb = b_ptr[i]; \
            d_ptr[i] = (TYPE_OUT)(EXPR); \
        } \
    } else { \
        mf_accessor_##ACC_OUT it_dst = mf_accessor_##ACC_OUT##_begin(dst); \
        mf_accessor_##ACC_IN it_a = mf_accessor_##ACC_IN##_begin(a); \
        mf_accessor_##ACC_IN it_b = mf_accessor_##ACC_IN##_begin(b); \
        for(size_t i=0; i<sz_dst; ++i) { \
            TYPE_IN va = mf_accessor_##ACC_IN##_get(&it_a); \
            TYPE_IN vb = mf_accessor_##ACC_IN##_get(&it_b); \
            mf_accessor_##ACC_OUT##_set(&it_dst, (TYPE_OUT)(EXPR)); \
            mf_accessor_##ACC_IN##_advance(&it_a, st1); \
            mf_accessor_##ACC_IN##_advance(&it_b, st2); \
            mf_accessor_##ACC_OUT##_advance(&it_dst, st0); \
        } \
    } \
}

#define MF_KERNEL_TERNARY_GENERIC(NAME, TYPE_A, TYPE_B, TYPE_C, TYPE_OUT, DTYPE_OUT, EXPR, ACC_IN, ACC_OUT) \
static void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    mf_tensor* dst = &ctx->registers[inst->dest_idx]; \
    mf_tensor* a = &ctx->registers[inst->src1_idx]; \
    mf_tensor* b = &ctx->registers[inst->src2_idx]; \
    mf_tensor* c = &ctx->registers[inst->src3_idx]; \
    MF_CHECK_DST_VIEW(ctx, dst); \
    MF_CHECK_INPUT(ctx, a); \
    MF_CHECK_INPUT(ctx, b); \
    MF_CHECK_INPUT(ctx, c); \
    dst->info.dtype = MF_DTYPE_##DTYPE_OUT; \
    if (!mf_utils_resolve_ternary_shape(ctx, dst, a, b, c)) return; \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    i32 st0 = MF_GET_STRIDE_D(inst); \
    i32 st1 = MF_GET_STRIDE_S1(inst); \
    i32 st2 = MF_GET_STRIDE_S2(inst); \
    i32 st3 = MF_GET_STRIDE_S3(inst); \
    if (st0 == 1 && st1 == 1 && st2 == 1 && st3 == 1 && dst->byte_offset == 0 && a->byte_offset == 0 && b->byte_offset == 0 && c->byte_offset == 0) { \
        TYPE_OUT* d_ptr = (TYPE_OUT*)dst->buffer->data; \
        TYPE_A* a_ptr = (TYPE_A*)a->buffer->data; \
        TYPE_B* b_ptr = (TYPE_B*)b->buffer->data; \
        TYPE_C* c_ptr = (TYPE_C*)c->buffer->data; \
        for(size_t i=0; i<sz_dst; ++i) { \
            TYPE_A va = a_ptr[i]; TYPE_B vb = b_ptr[i]; TYPE_C vc = c_ptr[i]; \
            d_ptr[i] = (TYPE_OUT)(EXPR); \
        } \
    } else { \
        mf_accessor_##ACC_OUT it_dst = mf_accessor_##ACC_OUT##_begin(dst); \
        mf_accessor_##ACC_IN it_a = mf_accessor_##ACC_IN##_begin(a); \
        mf_accessor_##ACC_IN it_b = mf_accessor_##ACC_IN##_begin(b); \
        mf_accessor_##ACC_IN it_c = mf_accessor_##ACC_IN##_begin(c); \
        for(size_t i=0; i<sz_dst; ++i) { \
            TYPE_A va = mf_accessor_##ACC_IN##_get(&it_a); \
            TYPE_B vb = mf_accessor_##ACC_IN##_get(&it_b); \
            TYPE_C vc = mf_accessor_##ACC_IN##_get(&it_c); \
            mf_accessor_##ACC_OUT##_set(&it_dst, (TYPE_OUT)(EXPR)); \
            mf_accessor_##ACC_IN##_advance(&it_a, st1); \
            mf_accessor_##ACC_IN##_advance(&it_b, st2); \
            mf_accessor_##ACC_IN##_advance(&it_c, st3); \
            mf_accessor_##ACC_OUT##_advance(&it_dst, st0); \
        } \
    } \
}

#define MF_KERNEL_UNARY_GENERIC(NAME, TYPE_IN, TYPE_OUT, DTYPE_OUT, EXPR, ACC_IN, ACC_OUT) \
static void op_##NAME(mf_exec_ctx* ctx, const struct mf_instruction* inst) { \
    mf_tensor* dst = &ctx->registers[inst->dest_idx]; \
    mf_tensor* a = &ctx->registers[inst->src1_idx]; \
    MF_CHECK_DST_VIEW(ctx, dst); \
    MF_CHECK_INPUT(ctx, a); \
    dst->info.dtype = MF_DTYPE_##DTYPE_OUT; \
    if (!mf_utils_resolve_unary_shape(ctx, dst, a)) return; \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    i32 st0 = MF_GET_STRIDE_D(inst); \
    i32 st1 = MF_GET_STRIDE_S1(inst); \
    if (st0 == 1 && st1 == 1 && dst->byte_offset == 0 && a->byte_offset == 0) { \
        TYPE_OUT* d_ptr = (TYPE_OUT*)dst->buffer->data; \
        TYPE_IN* a_ptr = (TYPE_IN*)a->buffer->data; \
        for(size_t i=0; i<sz_dst; ++i) { \
            TYPE_IN v = a_ptr[i]; \
            d_ptr[i] = (TYPE_OUT)(EXPR); \
        } \
    } else { \
        mf_accessor_##ACC_OUT it_dst = mf_accessor_##ACC_OUT##_begin(dst); \
        mf_accessor_##ACC_IN it_a = mf_accessor_##ACC_IN##_begin(a); \
        for(size_t i=0; i<sz_dst; ++i) { \
            TYPE_IN v = mf_accessor_##ACC_IN##_get(&it_a); \
            mf_accessor_##ACC_OUT##_set(&it_dst, (TYPE_OUT)(EXPR)); \
            mf_accessor_##ACC_IN##_advance(&it_a, st1); \
            mf_accessor_##ACC_OUT##_advance(&it_dst, st0); \
        } \
    } \
}

// --- Specific Shortcuts ---

#define MF_KERNEL_BINARY(NAME, OP) MF_KERNEL_BINARY_GENERIC(NAME, f32, f32, F32, MF_SAFE_F32(va OP vb), f32, f32)
#define MF_KERNEL_BINARY_FUNC(NAME, FUNC) MF_KERNEL_BINARY_GENERIC(NAME, f32, f32, F32, MF_SAFE_F32(FUNC(va, vb)), f32, f32)
#define MF_KERNEL_UNARY(NAME, FUNC) MF_KERNEL_UNARY_GENERIC(NAME, f32, f32, F32, MF_SAFE_F32(FUNC(v)), f32, f32)

#define MF_KERNEL_COMPARE(NAME, OP) MF_KERNEL_BINARY_GENERIC(NAME, f32, u8, U8, (va OP vb), f32, u8)
#define MF_KERNEL_LOGIC(NAME, OP) MF_KERNEL_BINARY_GENERIC(NAME, u8, u8, U8, (va OP vb), u8, u8)

#endif // MF_KERNEL_UTILS_H