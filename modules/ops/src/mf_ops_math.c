#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include "mf_ops_internal.h"
#include <math.h>

// Forward decls for clean types
typedef struct mf_tensor mf_tensor;
typedef struct mf_cpu_baked_instr mf_cpu_baked_instr;

// --- Arithmetic ---
MF_KERNEL_BINARY(add, +)
MF_KERNEL_BINARY(sub, -)
MF_KERNEL_BINARY(mul, *)
MF_KERNEL_BINARY(div, /)
MF_KERNEL_BINARY_FUNC(atan2, atan2f)
MF_KERNEL_BINARY_FUNC(pow, powf)

// --- Unary Math ---
MF_KERNEL_UNARY(sin, sinf)
MF_KERNEL_UNARY(cos, cosf)
MF_KERNEL_UNARY(floor, floorf)
MF_KERNEL_UNARY(ceil, ceilf)
MF_KERNEL_UNARY(abs, fabsf)
MF_KERNEL_UNARY(sqrt, sqrtf)

// --- Min/Max/Clamp/Mix ---

MF_KERNEL_BINARY_GENERIC(min, f32, f32, F32, (va < vb ? va : vb), f32, f32)

MF_KERNEL_BINARY_GENERIC(max, f32, f32, F32, (va > vb ? va : vb), f32, f32)

MF_KERNEL_TERNARY_GENERIC(fma, f32, f32, f32, f32, F32, MF_SAFE_F32(fmaf(va, vb, vc)), f32, f32)

MF_KERNEL_TERNARY_GENERIC(clamp, f32, f32, f32, f32, F32, MF_SAFE_F32(fminf(fmaxf(va, vb), vc)), f32, f32)

MF_KERNEL_TERNARY_GENERIC(mix, f32, f32, f32, f32, F32, MF_SAFE_F32(va * (1.0f - vc) + vb * vc), f32, f32)

static void op_smoothstep(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* e = bi->s1; // Edges [2] or Scalar
    mf_tensor* x = bi->s2;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, e);
    MF_CHECK_INPUT(ctx, x);

    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_utils_resolve_unary_shape(ctx, dst, x)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    size_t sz = mf_tensor_count(dst);
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);
    mf_accessor_f32 it_x = mf_accessor_f32_begin(x);
    mf_accessor_f32 it_e = mf_accessor_f32_begin(e);

    // Smoothstep has 2 edges. If e has 2 elements, we use them. If 1, we assume [0, e].
    f32 e0 = 0.0f;
    f32 e1 = 1.0f;
    
    if (mf_tensor_count(e) >= 2) {
        e0 = mf_accessor_f32_get(&it_e);
        mf_accessor_f32_advance(&it_e, 1);
        e1 = mf_accessor_f32_get(&it_e);
    } else if (mf_tensor_count(e) == 1) {
        e1 = mf_accessor_f32_get(&it_e);
    }

    f32 span = e1 - e0;
    if (fabsf(span) < 1e-6f) span = 1e-6f;

    i32 st0 = MF_GET_STRIDE(dst);
    i32 st2 = MF_GET_STRIDE(x);

    for (size_t i = 0; i < sz; ++i) {
        f32 val = mf_accessor_f32_get(&it_x);
        f32 t = (val - e0) / span;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        f32 res = t * t * (3.0f - 2.0f * t);
        mf_accessor_f32_set(&it_dst, MF_SAFE_F32(res));
        
        mf_accessor_f32_advance(&it_x, st2);
        mf_accessor_f32_advance(&it_dst, st0);
    }
}

static inline f32 _vec_dot(mf_accessor_f32* it_a, mf_accessor_f32* it_b, size_t len) {
    f32 sum = 0;
    for (size_t j = 0; j < len; ++j) {
        sum += mf_accessor_f32_get(it_a) * mf_accessor_f32_get(it_b);
        mf_accessor_f32_advance(it_a, 1);
        mf_accessor_f32_advance(it_b, 1);
    }
    return sum;
}

static inline f32 _vec_len_sq(mf_accessor_f32* it_a, size_t len) {
    f32 sum = 0;
    for (size_t j = 0; j < len; ++j) {
        f32 v = mf_accessor_f32_get(it_a);
        sum += v * v;
        mf_accessor_f32_advance(it_a, 1);
    }
    return sum;
}

static void op_dot(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* a = bi->s1;
    mf_tensor* b = bi->s2;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    
    dst->info.dtype = MF_DTYPE_F32;
    if (a->info.ndim > 1) {
        if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim - 1)) return;
    } else {
        u32 scalar_shape[] = {1};
        if (!mf_exec_ctx_resize_tensor(ctx, dst, scalar_shape, 1)) return;
    }
    MF_CHECK_DST_DATA(ctx, dst);

    size_t vec_len = a->info.shape[a->info.ndim - 1];
    size_t out_count = mf_tensor_count(dst);
    
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);
    mf_accessor_f32 it_a = mf_accessor_f32_begin(a);
    mf_accessor_f32 it_b = mf_accessor_f32_begin(b);
    i32 st0 = MF_GET_STRIDE(dst);

    for (size_t i = 0; i < out_count; ++i) {
        mf_accessor_f32_set(&it_dst, MF_SAFE_F32(_vec_dot(&it_a, &it_b, vec_len)));
        mf_accessor_f32_advance(&it_dst, st0);
    }
}

static void op_length(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* a = bi->s1;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    
    dst->info.dtype = MF_DTYPE_F32;
    if (a->info.ndim > 1) {
        if (!mf_exec_ctx_resize_tensor(ctx, dst, a->info.shape, a->info.ndim - 1)) return;
    } else {
        u32 scalar_shape[] = {1};
        if (!mf_exec_ctx_resize_tensor(ctx, dst, scalar_shape, 1)) return;
    }
    MF_CHECK_DST_DATA(ctx, dst);

    size_t vec_len = a->info.shape[a->info.ndim - 1];
    size_t out_count = mf_tensor_count(dst);
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);
    mf_accessor_f32 it_a = mf_accessor_f32_begin(a);
    i32 st0 = MF_GET_STRIDE(dst);

    for (size_t i = 0; i < out_count; ++i) {
        mf_accessor_f32_set(&it_dst, MF_SAFE_F32(sqrtf(_vec_len_sq(&it_a, vec_len))));
        mf_accessor_f32_advance(&it_dst, st0);
    }
}

static void op_normalize(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* a = bi->s1;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    
    dst->info.dtype = MF_DTYPE_F32;
    if (!mf_utils_resolve_unary_shape(ctx, dst, a)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    size_t vec_len = a->info.shape[a->info.ndim - 1];
    size_t out_count = mf_tensor_count(dst) / vec_len;
    
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);
    mf_accessor_f32 it_a = mf_accessor_f32_begin(a);

    for (size_t i = 0; i < out_count; ++i) {
        mf_accessor_f32 it_calc = it_a;
        f32 len = sqrtf(_vec_len_sq(&it_calc, vec_len));
        f32 inv_len = (len > 1e-6f) ? (1.0f / len) : 0.0f;

        for (size_t j = 0; j < vec_len; ++j) {
            mf_accessor_f32_set(&it_dst, mf_accessor_f32_get(&it_a) * inv_len);
            mf_accessor_f32_advance(&it_a, 1);
            mf_accessor_f32_advance(&it_dst, 1);
        }
    }
}

// --- Reduction ---

static void op_reduce_sum(mf_exec_ctx* ctx, const mf_cpu_baked_instr* bi) {
    mf_tensor* dst = bi->d;
    mf_tensor* src = bi->s1;
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, src);
    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t count = (ctx->batch_size > 0) ? ctx->batch_size : mf_tensor_count(src);
    f32 sum = 0;
    
    mf_accessor_f32 it = mf_accessor_f32_begin(src);
    i32 st1 = MF_GET_STRIDE(src);

    for (size_t i = 0; i < count; ++i) {
        sum += mf_accessor_f32_get(&it);
        mf_accessor_f32_advance(&it, st1);
    }
    
    // Result is a scalar
    mf_accessor_f32 out = mf_accessor_f32_begin(dst);
    mf_accessor_f32_set(&out, sum);
}

void mf_ops_register_math(mf_op_func* table) {
    table[MF_OP_ADD] = op_add;
    table[MF_OP_SUB] = op_sub;
    table[MF_OP_MUL] = op_mul;
    table[MF_OP_DIV] = op_div;
    table[MF_OP_SIN] = op_sin;
    table[MF_OP_COS] = op_cos;
    table[MF_OP_FLOOR] = op_floor;
    table[MF_OP_CEIL] = op_ceil;
    table[MF_OP_ABS] = op_abs;
    table[MF_OP_SQRT] = op_sqrt;
    table[MF_OP_MIN] = op_min;
    table[MF_OP_MAX] = op_max;
    table[MF_OP_FMA] = op_fma;
    table[MF_OP_CLAMP] = op_clamp;
    table[MF_OP_MIX] = op_mix;
    table[MF_OP_SMOOTHSTEP] = op_smoothstep;
    table[MF_OP_DOT] = op_dot;
    table[MF_OP_LENGTH] = op_length;
    table[MF_OP_NORMALIZE] = op_normalize;
    table[MF_OP_POW] = op_pow;
    table[MF_OP_ATAN2] = op_atan2;
    table[MF_OP_SUM] = op_reduce_sum;
}
