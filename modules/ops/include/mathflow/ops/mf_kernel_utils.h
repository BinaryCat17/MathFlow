#ifndef MF_KERNEL_UTILS_H
#define MF_KERNEL_UTILS_H

#include <mathflow/isa/mf_kernel_ctx.h>
#include <mathflow/isa/mf_tensor.h>
#include <math.h>
#include <string.h>

// --- Helper: Shape Resolution (Inline) ---

static inline bool mf_utils_resolve_binary_shape(const mf_kernel_ctx* ctx, mf_tensor* dst, const mf_tensor* a, const mf_tensor* b) {
    if (dst->dtype == MF_DTYPE_UNKNOWN) dst->dtype = a->dtype;

    // Virtual Batching
    if (ctx->batch_size > 0) {
        // If batching is active, we force the output to be a 1D vector of batch_size
        // UNLESS both inputs are scalars (size=1), then output is scalar.
        bool a_scalar = (a->size == 1);
        bool b_scalar = (b->size == 1);
        
        if (a_scalar && b_scalar) {
             int32_t scalar_shape[] = {1};
             return ctx->resize_tensor(ctx->impl, dst, scalar_shape, 0);
        } else {
             int32_t batch_shape[] = { (int32_t)ctx->batch_size };
             return ctx->resize_tensor(ctx->impl, dst, batch_shape, 1);
        }
    }

    const mf_tensor* shape_src = (a->size >= b->size) ? a : b;
    return ctx->resize_tensor(ctx->impl, dst, shape_src->shape, shape_src->ndim);
}

static inline bool mf_utils_resolve_unary_shape(const mf_kernel_ctx* ctx, mf_tensor* dst, const mf_tensor* a) {
    if (dst->dtype == MF_DTYPE_UNKNOWN) dst->dtype = a->dtype;
    
    // Virtual Batching
    if (ctx->batch_size > 0) {
        if (a->size == 1) {
             int32_t scalar_shape[] = {1};
             return ctx->resize_tensor(ctx->impl, dst, scalar_shape, 0);
        } else {
             int32_t batch_shape[] = { (int32_t)ctx->batch_size };
             return ctx->resize_tensor(ctx->impl, dst, batch_shape, 1);
        }
    }

    return ctx->resize_tensor(ctx->impl, dst, a->shape, a->ndim);
}

// --- Macros: Kernel Definitions ---

// Helper for generic binary ops (C = A op B)
// Supports scalar broadcasting (if size==1)
#define MF_KERNEL_BINARY(NAME, OP) \
static void op_##NAME(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    size_t sz_a = a->size; size_t sz_b = b->size; \
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = da[i % sz_a] OP db[i % sz_b]; \
}

// Helper for function-based binary ops (C = func(A, B))
#define MF_KERNEL_BINARY_FUNC(NAME, FUNC) \
static void op_##NAME(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    size_t sz_a = a->size; size_t sz_b = b->size; \
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = FUNC(da[i % sz_a], db[i % sz_b]); \
}

// Helper for unary ops (C = func(A))
#define MF_KERNEL_UNARY(NAME, FUNC) \
static void op_##NAME(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ); \
    if (!dst || !a) return; \
    if (!mf_utils_resolve_unary_shape(ctx, dst, a)) return; \
    f32* da = (f32*)a->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = FUNC(da[i]); \
}

// Helper for comparison ops (C = A op B), Output is always U8 (bool)
#define MF_KERNEL_COMPARE(NAME, OP) \
static void op_##NAME(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->dtype = MF_DTYPE_U8; /* Force Bool Output */ \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    size_t sz_a = a->size; size_t sz_b = b->size; u8* dd = (u8*)dst->data; \
    if (a->dtype == MF_DTYPE_F32) { \
        f32* da = (f32*)a->data; f32* db = (f32*)b->data; \
        for(size_t i=0; i<dst->size; ++i) dd[i] = (da[i % sz_a] OP db[i % sz_b]) ? 1 : 0; \
    } else if (a->dtype == MF_DTYPE_I32) { \
        int32_t* da = (int32_t*)a->data; int32_t* db = (int32_t*)b->data; \
        for(size_t i=0; i<dst->size; ++i) dd[i] = (da[i % sz_a] OP db[i % sz_b]) ? 1 : 0; \
    } \
}

// Helper for logic ops (C = A op B), Input/Output U8
#define MF_KERNEL_LOGIC(NAME, OP) \
static void op_##NAME(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->dtype = MF_DTYPE_U8; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    u8* da = (u8*)a->data; u8* db = (u8*)b->data; u8* dd = (u8*)dst->data; \
    size_t sz_a = a->size; size_t sz_b = b->size; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = da[i % sz_a] OP db[i % sz_b]; \
}

#endif // MF_KERNEL_UTILS_H
