#ifndef MF_KERNEL_UTILS_H
#define MF_KERNEL_UTILS_H

#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_instruction.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <math.h>
#include <mathflow/isa/mf_tensor_iter.h>
#include <string.h>

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

// --- Macros: Optimized Kernel Definitions ---

#define MF_KERNEL_BINARY_GENERIC(NAME, TYPE_IN, TYPE_OUT, DTYPE_OUT, EXPR) \
static void op_##NAME(mf_exec_ctx* ctx, const mf_instruction* inst) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ); \
    MF_CHECK_DST_VIEW(ctx, dst); \
    MF_CHECK_INPUT(ctx, a); \
    MF_CHECK_INPUT(ctx, b); \
    dst->info.dtype = MF_DTYPE_##DTYPE_OUT; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); \
    size_t sz_dst = mf_tensor_count(dst); \
    mf_tensor_iter it_dst = mf_tensor_iter_begin(dst); \
    mf_tensor_iter it_a = mf_tensor_iter_begin(a); \
    mf_tensor_iter it_b = mf_tensor_iter_begin(b); \
    for(size_t i=0; i<sz_dst; ++i) { \
        TYPE_IN va = *((TYPE_IN*)it_a.ptr); \
        TYPE_IN vb = *((TYPE_IN*)it_b.ptr); \
        *((TYPE_OUT*)it_dst.ptr) = (TYPE_OUT)(EXPR); \
        mf_tensor_iter_advance(&it_a, inst->strides[1]); \
        mf_tensor_iter_advance(&it_b, inst->strides[2]); \
        mf_tensor_iter_advance(&it_dst, inst->strides[0]); \
    } \
}

#define MF_KERNEL_TERNARY_GENERIC(NAME, TYPE_A, TYPE_B, TYPE_C, TYPE_OUT, DTYPE_OUT, EXPR) \
static void op_##NAME(mf_exec_ctx* ctx, const mf_instruction* inst) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ); \
    mf_tensor* c = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ); \
    MF_CHECK_DST_VIEW(ctx, dst); \
    MF_CHECK_INPUT(ctx, a); \
    MF_CHECK_INPUT(ctx, b); \
    MF_CHECK_INPUT(ctx, c); \
    dst->info.dtype = MF_DTYPE_##DTYPE_OUT; \
    if (!mf_utils_resolve_ternary_shape(ctx, dst, a, b, c)) return; \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); \
    size_t sz_c = mf_tensor_count(c); size_t sz_dst = mf_tensor_count(dst); \
    mf_tensor_iter it_dst = mf_tensor_iter_begin(dst); \
    mf_tensor_iter it_a = mf_tensor_iter_begin(a); \
    mf_tensor_iter it_b = mf_tensor_iter_begin(b); \
    mf_tensor_iter it_c = mf_tensor_iter_begin(c); \
    for(size_t i=0; i<sz_dst; ++i) { \
        TYPE_A va = *((TYPE_A*)it_a.ptr); \
        TYPE_B vb = *((TYPE_B*)it_b.ptr); \
        TYPE_C vc = *((TYPE_C*)it_c.ptr); \
        *((TYPE_OUT*)it_dst.ptr) = (TYPE_OUT)(EXPR); \
        mf_tensor_iter_advance(&it_a, inst->strides[1]); \
        mf_tensor_iter_advance(&it_b, inst->strides[2]); \
        mf_tensor_iter_advance(&it_c, inst->strides[3]); \
        mf_tensor_iter_advance(&it_dst, inst->strides[0]); \
    } \
}

#define MF_KERNEL_UNARY_GENERIC(NAME, TYPE_IN, TYPE_OUT, DTYPE_OUT, EXPR) \
static void op_##NAME(mf_exec_ctx* ctx, const mf_instruction* inst) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ); \
    MF_CHECK_DST_VIEW(ctx, dst); \
    MF_CHECK_INPUT(ctx, a); \
    dst->info.dtype = MF_DTYPE_##DTYPE_OUT; \
    if (!mf_utils_resolve_unary_shape(ctx, dst, a)) return; \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    mf_tensor_iter it_dst = mf_tensor_iter_begin(dst); \
    mf_tensor_iter it_a = mf_tensor_iter_begin(a); \
    for(size_t i=0; i<sz_dst; ++i) { \
        TYPE_IN v = *((TYPE_IN*)it_a.ptr); \
        *((TYPE_OUT*)it_dst.ptr) = (TYPE_OUT)(EXPR); \
        mf_tensor_iter_advance(&it_a, inst->strides[1]); \
        mf_tensor_iter_advance(&it_dst, inst->strides[0]); \
    } \
}

// --- Specific Shortcuts ---

#define MF_KERNEL_BINARY(NAME, OP) MF_KERNEL_BINARY_GENERIC(NAME, f32, f32, F32, (va OP vb))
#define MF_KERNEL_BINARY_FUNC(NAME, FUNC) MF_KERNEL_BINARY_GENERIC(NAME, f32, f32, F32, FUNC(va, vb))
#define MF_KERNEL_UNARY(NAME, FUNC) MF_KERNEL_UNARY_GENERIC(NAME, f32, f32, F32, FUNC(v))

#define MF_KERNEL_COMPARE(NAME, OP) MF_KERNEL_BINARY_GENERIC(NAME, f32, u8, U8, (va OP vb))
#define MF_KERNEL_LOGIC(NAME, OP) MF_KERNEL_BINARY_GENERIC(NAME, u8, u8, U8, (va OP vb))

#endif // MF_KERNEL_UTILS_H
