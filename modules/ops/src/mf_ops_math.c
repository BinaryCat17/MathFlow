#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include "mf_ops_internal.h"
#include <math.h>

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
MF_KERNEL_BINARY_GENERIC(min, f32, f32, F32, (va < vb ? va : vb))
MF_KERNEL_BINARY_GENERIC(max, f32, f32, F32, (va > vb ? va : vb))

static void op_clamp(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* v = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* min_val = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    mf_tensor* max_val = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, v);
    MF_CHECK_INPUT(ctx, min_val);
    MF_CHECK_INPUT(ctx, max_val);
    
    if (!mf_utils_resolve_ternary_shape(ctx, dst, v, min_val, max_val)) return;
    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t sz_v = mf_tensor_count(v);
    size_t sz_min = mf_tensor_count(min_val);
    size_t sz_max = mf_tensor_count(max_val);
    size_t sz_dst = mf_tensor_count(dst);
    
    f32* dv = (f32*)mf_tensor_data(v);
    f32* dmin = (f32*)mf_tensor_data(min_val);
    f32* dmax = (f32*)mf_tensor_data(max_val);
    f32* dd = (f32*)mf_tensor_data(dst);
    
    for(size_t i=0; i<sz_dst; ++i) {
        f32 val = dv[i % sz_v];
        f32 lo = dmin[i % sz_min];
        f32 hi = dmax[i % sz_max];
        dd[i] = (val < lo) ? lo : (val > hi ? hi : val);
    }
}

static void op_mix(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    mf_tensor* t = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    MF_CHECK_INPUT(ctx, t);
    
    if (!mf_utils_resolve_ternary_shape(ctx, dst, a, b, t)) return;
    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t sz_a = mf_tensor_count(a);
    size_t sz_b = mf_tensor_count(b);
    size_t sz_t = mf_tensor_count(t);
    size_t sz_dst = mf_tensor_count(dst);
    
    f32* da = (f32*)mf_tensor_data(a);
    f32* db = (f32*)mf_tensor_data(b);
    f32* dt = (f32*)mf_tensor_data(t);
    f32* dd = (f32*)mf_tensor_data(dst);
    
    for(size_t i=0; i<sz_dst; ++i) {
        f32 va = da[i % sz_a];
        f32 vb = db[i % sz_b];
        f32 vt = dt[i % sz_t];
        dd[i] = va + vt * (vb - va);
    }
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
    table[MF_OP_CLAMP] = op_clamp;
    table[MF_OP_MIX] = op_mix;
    table[MF_OP_POW] = op_pow;
    table[MF_OP_ATAN2] = op_atan2;
}
