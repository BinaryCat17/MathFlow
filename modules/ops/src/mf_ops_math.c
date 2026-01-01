#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include "mf_ops_internal.h"
#include <math.h>

// --- Macros: Kernel Definitions (Optimized) ---

#undef MF_KERNEL_BINARY
#define MF_KERNEL_BINARY(NAME, OP) \
static void op_##NAME(mf_exec_ctx* ctx, const mf_instruction* inst) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ); \
    MF_CHECK_DST_VIEW(ctx, dst); \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    MF_CHECK_INPUT(ctx, a); \
    MF_CHECK_INPUT(ctx, b); \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); \
    f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); f32* dd = (f32*)mf_tensor_data(dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    bool a_s = (sz_a == 1); bool b_s = (sz_b == 1); \
    for(size_t i=0; i<sz_dst; ++i) dd[i] = (a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i]); \
}

#undef MF_KERNEL_BINARY_FUNC
#define MF_KERNEL_BINARY_FUNC(NAME, FUNC) \
static void op_##NAME(mf_exec_ctx* ctx, const mf_instruction* inst) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ); \
    MF_CHECK_DST_VIEW(ctx, dst); \
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return; \
    MF_CHECK_INPUT(ctx, a); \
    MF_CHECK_INPUT(ctx, b); \
    MF_CHECK_DST_DATA(ctx, dst); \
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); \
    f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); f32* dd = (f32*)mf_tensor_data(dst); \
    size_t sz_dst = mf_tensor_count(dst); \
    bool a_s = (sz_a == 1); bool b_s = (sz_b == 1); \
    for(size_t i=0; i<sz_dst; ++i) dd[i] = FUNC(a_s ? da[0] : da[i], b_s ? db[0] : db[i]); \
}

// --- Arithmetic ---

MF_KERNEL_BINARY(add, +)
MF_KERNEL_BINARY(sub, -)
MF_KERNEL_BINARY(mul, *)
MF_KERNEL_BINARY(div, /)
MF_KERNEL_BINARY_FUNC(atan2, atan2f)
MF_KERNEL_BINARY_FUNC(pow, powf)

// --- Unary Math ---

#undef MF_KERNEL_UNARY
#define MF_KERNEL_UNARY(NAME, FUNC) \
static void op_##NAME(mf_exec_ctx* ctx, const mf_instruction* inst) { \
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE); \
    mf_tensor* src = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ); \
    MF_CHECK_DST_VIEW(ctx, dst); \
    if (!mf_utils_resolve_unary_shape(ctx, dst, src)) return; \
    MF_CHECK_INPUT(ctx, src); \
    MF_CHECK_DST_DATA(ctx, dst); \
    f32* s = (f32*)mf_tensor_data(src); f32* d = (f32*)mf_tensor_data(dst); \
    size_t sz = mf_tensor_count(dst); \
    for(size_t i=0; i<sz; ++i) d[i] = FUNC(s[i]); \
}

MF_KERNEL_UNARY(sin, sinf)
MF_KERNEL_UNARY(cos, cosf)
MF_KERNEL_UNARY(floor, floorf)
MF_KERNEL_UNARY(ceil, ceilf)
MF_KERNEL_UNARY(abs, fabsf)
MF_KERNEL_UNARY(sqrt, sqrtf)

// --- Min/Max/Clamp/Mix ---

static void op_min(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    MF_CHECK_DST_VIEW(ctx, dst);
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return;
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    MF_CHECK_DST_DATA(ctx, dst);
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b);
    f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); f32* dd = (f32*)mf_tensor_data(dst);
    size_t sz_dst = mf_tensor_count(dst);
    bool a_s = (sz_a == 1); bool b_s = (sz_b == 1);
    for(size_t i=0; i<sz_dst; ++i) { 
        f32 va = a_s ? da[0] : da[i]; f32 vb = b_s ? db[0] : db[i];
        dd[i] = (va < vb) ? va : vb; 
    }
}

static void op_max(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    MF_CHECK_DST_VIEW(ctx, dst);
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return;
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    MF_CHECK_DST_DATA(ctx, dst);
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b);
    f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); f32* dd = (f32*)mf_tensor_data(dst);
    size_t sz_dst = mf_tensor_count(dst);
    bool a_s = (sz_a == 1); bool b_s = (sz_b == 1);
    for(size_t i=0; i<sz_dst; ++i) { 
        f32 va = a_s ? da[0] : da[i]; f32 vb = b_s ? db[0] : db[i];
        dd[i] = (va > vb) ? va : vb; 
    }
}

static void op_mix(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    mf_tensor* t = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ);
    MF_CHECK_DST_VIEW(ctx, dst);
    if (!mf_utils_resolve_ternary_shape(ctx, dst, a, b, t)) return;
    MF_CHECK_INPUT(ctx, a);
    MF_CHECK_INPUT(ctx, b);
    MF_CHECK_INPUT(ctx, t);
    MF_CHECK_DST_DATA(ctx, dst);
    size_t sz_a = mf_tensor_count(a); size_t sz_b = mf_tensor_count(b); size_t sz_t = mf_tensor_count(t);
    f32* da = (f32*)mf_tensor_data(a); f32* db = (f32*)mf_tensor_data(b); f32* dt = (f32*)mf_tensor_data(t); f32* dd = (f32*)mf_tensor_data(dst);
    size_t sz_dst = mf_tensor_count(dst);
    bool a_s = (sz_a == 1); bool b_s = (sz_b == 1); bool t_s = (sz_t == 1);
    for(size_t i=0; i<sz_dst; ++i) { 
        f32 va = a_s ? da[0] : da[i]; f32 vb = b_s ? db[0] : db[i]; f32 vt = t_s ? dt[0] : dt[i];
        dd[i] = va * (1.0f - vt) + vb * vt;
    }
}

static void op_clamp(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* val = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* min_v = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    mf_tensor* max_v = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ);
    MF_CHECK_DST_VIEW(ctx, dst);
    if (!mf_utils_resolve_ternary_shape(ctx, dst, val, min_v, max_v)) return;
    MF_CHECK_INPUT(ctx, val);
    MF_CHECK_INPUT(ctx, min_v);
    MF_CHECK_INPUT(ctx, max_v);
    MF_CHECK_DST_DATA(ctx, dst);
    size_t sz_val = mf_tensor_count(val); size_t sz_min = mf_tensor_count(min_v); size_t sz_max = mf_tensor_count(max_v);
    f32* d_val = (f32*)mf_tensor_data(val); f32* d_min = (f32*)mf_tensor_data(min_v); f32* d_max = (f32*)mf_tensor_data(max_v); f32* d_dst = (f32*)mf_tensor_data(dst);
    size_t sz_dst = mf_tensor_count(dst);
    bool val_s = (sz_val == 1); bool min_s = (sz_min == 1); bool max_s = (sz_max == 1);
    for(size_t i=0; i<sz_dst; ++i) { 
        f32 v = val_s ? d_val[0] : d_val[i]; f32 mn = min_s ? d_min[0] : d_min[i]; f32 mx = max_s ? d_max[0] : d_max[i];
        if (v < mn) v = mn; if (v > mx) v = mx;
        d_dst[i] = v;
    }
}

// --- GLSL Helpers ---

static void op_smoothstep(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* val = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* edges = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    MF_CHECK_DST_VIEW(ctx, dst);
    if (!mf_utils_resolve_unary_shape(ctx, dst, val)) return; 
    MF_CHECK_INPUT(ctx, val);
    MF_CHECK_INPUT(ctx, edges);
    MF_CHECK_DST_DATA(ctx, dst);
    dst->info.dtype = MF_DTYPE_F32;
    f32* X = (f32*)mf_tensor_data(val); f32* E = (f32*)mf_tensor_data(edges); f32* D = (f32*)mf_tensor_data(dst);
    size_t sz_edges = mf_tensor_count(edges);
    bool uniform_edges = (sz_edges == 2);
    size_t sz_dst = mf_tensor_count(dst);
    for (size_t i = 0; i < sz_dst; ++i) {
        float x = X[i];
        float e0 = uniform_edges ? E[0] : E[i*2 + 0];
        float e1 = uniform_edges ? E[1] : E[i*2 + 1];
        float t = (x - e0) / (e1 - e0);
        if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
        D[i] = t * t * (3.0f - 2.0f * t);
    }
}

static void op_step(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* edge = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* x = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    MF_CHECK_DST_VIEW(ctx, dst);
    if (!mf_utils_resolve_binary_shape(ctx, dst, edge, x)) return;
    MF_CHECK_INPUT(ctx, edge);
    MF_CHECK_INPUT(ctx, x);
    MF_CHECK_DST_DATA(ctx, dst);
    dst->info.dtype = MF_DTYPE_F32;
    size_t sz_edge = mf_tensor_count(edge); size_t sz_x = mf_tensor_count(x);
    f32* de = (f32*)mf_tensor_data(edge); f32* dx = (f32*)mf_tensor_data(x); f32* dd = (f32*)mf_tensor_data(dst);
    size_t sz_dst = mf_tensor_count(dst);
    bool e_s = (sz_edge == 1); bool x_s = (sz_x == 1);
    for(size_t i=0; i<sz_dst; ++i) {
        f32 e_val = e_s ? de[0] : de[i]; f32 x_val = x_s ? dx[0] : dx[i];
        dd[i] = (x_val >= e_val) ? 1.0f : 0.0f;
    }
}

void mf_ops_register_math(mf_op_func* table) {
    table[MF_OP_ADD] = op_add; 
    table[MF_OP_SUB] = op_sub; 
    table[MF_OP_MUL] = op_mul; 
    table[MF_OP_DIV] = op_div;
    table[MF_OP_MIX] = op_mix;
    table[MF_OP_SIN] = op_sin; 
    table[MF_OP_COS] = op_cos; 
    table[MF_OP_FLOOR] = op_floor; 
    table[MF_OP_CEIL] = op_ceil; 
    table[MF_OP_ABS] = op_abs; 
    table[MF_OP_SQRT] = op_sqrt; 
    table[MF_OP_ATAN2] = op_atan2; 
    table[MF_OP_POW] = op_pow; 
    table[MF_OP_MIN] = op_min; 
    table[MF_OP_MAX] = op_max; 
    table[MF_OP_CLAMP] = op_clamp;
    table[MF_OP_SMOOTHSTEP] = op_smoothstep; 
    table[MF_OP_STEP] = op_step; 
}