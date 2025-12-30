#ifndef MF_KERNEL_UTILS_H
#define MF_KERNEL_UTILS_H

#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/isa/mf_tensor.h>
#include <math.h>
#include <string.h>

// --- Helper: Shape Resolution (Inline) ---

static inline bool mf_utils_resolve_binary_shape(mf_exec_ctx* ctx, mf_tensor* dst, const mf_tensor* a, const mf_tensor* b) {
    if (dst->info.dtype == MF_DTYPE_UNKNOWN) dst->info.dtype = a->info.dtype;

    // Use larger input as shape source (simple broadcasting)
    size_t count_a = mf_tensor_count(a);
    size_t count_b = mf_tensor_count(b);
    const mf_tensor* shape_src = (count_a >= count_b) ? a : b;
    return mf_exec_ctx_resize_tensor(ctx, dst, shape_src->info.shape, shape_src->info.ndim);
}

static inline bool mf_utils_resolve_unary_shape(mf_exec_ctx* ctx, mf_tensor* dst, const mf_tensor* a) {
    if (dst->info.dtype == MF_DTYPE_UNKNOWN) dst->info.dtype = a->info.dtype;
    return mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim);
}

// --- Macros: Kernel Definitions ---

// Helper for generic binary ops (C = A op B)
// Supports scalar broadcasting (if size==1)
#define MF_KERNEL_BINARY(NAME, OP) \
static void op_##NAME(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); \
    f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); f32* dd = (f32*)mf_tensor_data(dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    for(size_t i=0; i<sz_dst; ++i) dd[i] = da[i % sz_a] OP db[i % sz_b]; \
}

// Helper for function-based binary ops (C = func(A, B))
#define MF_KERNEL_BINARY_FUNC(NAME, FUNC) \
static void op_##NAME(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); \
    f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); f32* dd = (f32*)mf_tensor_data(dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    for(size_t i=0; i<sz_dst; ++i) dd[i] = FUNC(da[i % sz_a], db[i % sz_b]); \
}

// Helper for unary ops (C = func(A))
#define MF_KERNEL_UNARY(NAME, FUNC) \
static void op_##NAME(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ); \
    if (!dst || !a) return; \
    if (!mf_utils_resolve_unary_shape(ctx, dst, a)) return; \
    f32* da = (f32*)mf_tensor_data(a); f32* dd = (f32*)mf_tensor_data(dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    for(size_t i=0; i<sz_dst; ++i) dd[i] = FUNC(da[i]); \
}

// Helper for comparison ops (C = A op B), Output is always U8 (bool)
#define MF_KERNEL_COMPARE(NAME, OP) \
static void op_##NAME(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->info.dtype = MF_DTYPE_U8; /* Force Bool Output */ \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); u8* dd = (u8*)mf_tensor_data(dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    if (a->info.dtype == MF_DTYPE_F32) { \
        f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); \
        for(size_t i=0; i<sz_dst; ++i) dd[i] = (da[i % sz_a] OP db[i % sz_b]) ? 1 : 0; \
    } else if (a->info.dtype == MF_DTYPE_I32) { \
        int32_t* da = (int32_t*)mf_tensor_data(a); int32_t* db = (int32_t*)mf_tensor_data(b); \
        for(size_t i=0; i<sz_dst; ++i) dd[i] = (da[i % sz_a] OP db[i % sz_b]) ? 1 : 0; \
    } \
}

// Helper for logic ops (C = A op B), Input/Output U8
#define MF_KERNEL_LOGIC(NAME, OP) \
static void op_##NAME(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->info.dtype = MF_DTYPE_U8; \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    u8* da = (u8*)mf_tensor_data(a); u8* db = (u8*)mf_tensor_data(b); u8* dd = (u8*)mf_tensor_data(dst); \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); \
    size_t sz_dst = mf_tensor_count(dst); \
    for(size_t i=0; i<sz_dst; ++i) dd[i] = da[i % sz_a] OP db[i % sz_b]; \
}

#endif // MF_KERNEL_UTILS_H
