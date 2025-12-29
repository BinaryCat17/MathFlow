#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/ops/mf_kernel_utils.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <math.h>

// --- Arithmetic ---

static void op_add(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return;
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data;
    bool a_s = (a->size == 1); bool b_s = (b->size == 1);
    for(size_t i=0; i<dst->size; ++i) { 
        f32 va = a_s ? da[0] : da[i]; 
        f32 vb = b_s ? db[0] : db[i]; 
        dd[i] = va + vb; 
    }
}

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

// --- Min/Max ---

static void op_min(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return;
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data;
    bool a_s = (a->size == 1); bool b_s = (b->size == 1);
    for(size_t i=0; i<dst->size; ++i) { 
        f32 va = a_s ? da[0] : da[i]; 
        f32 vb = b_s ? db[0] : db[i]; 
        dd[i] = (va < vb) ? va : vb; 
    }
}

static void op_max(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ);
    if (!dst || !a || !b) return;
    if (!mf_utils_resolve_binary_shape(ctx, dst, a, b)) return;
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data;
    bool a_s = (a->size == 1); bool b_s = (b->size == 1);
    for(size_t i=0; i<dst->size; ++i) { 
        f32 va = a_s ? da[0] : da[i]; 
        f32 vb = b_s ? db[0] : db[i]; 
        dd[i] = (va > vb) ? va : vb; 
    }
}

// --- GLSL Helpers ---

static void op_smoothstep(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* val = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    mf_tensor* edges = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ);
    if (!dst || !val || !edges) return;
    
    if (!mf_utils_resolve_unary_shape(ctx, dst, val)) return; 
    dst->dtype = MF_DTYPE_F32;

    f32* X = (f32*)val->data;
    f32* E = (f32*)edges->data;
    f32* D = (f32*)dst->data;
    
    bool uniform_edges = (edges->size == 2);
    
    for (size_t i = 0; i < dst->size; ++i) {
        float x = X[i];
        float e0 = uniform_edges ? E[0] : E[i*2 + 0];
        float e1 = uniform_edges ? E[1] : E[i*2 + 1];
        
        float t = (x - e0) / (e1 - e0);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        D[i] = t * t * (3.0f - 2.0f * t);
    }
}

static void op_step(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* edge = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    mf_tensor* x = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ);
    if (!dst || !edge || !x) return;
    
    if (!mf_utils_resolve_binary_shape(ctx, dst, edge, x)) return;
    dst->dtype = MF_DTYPE_F32;
    
    f32* de = (f32*)edge->data; f32* dx = (f32*)x->data; f32* dd = (f32*)dst->data;
    bool e_s = (edge->size == 1); bool x_s = (x->size == 1);
    
    for(size_t i=0; i<dst->size; ++i) {
        f32 e_val = e_s ? de[0] : de[i];
        f32 x_val = x_s ? dx[0] : dx[i];
        dd[i] = (x_val >= e_val) ? 1.0f : 0.0f;
    }
}

void mf_ops_register_math(mf_backend_dispatch_table* table) {
    table->op_table[MF_OP_ADD] = op_add; 
    table->op_table[MF_OP_SUB] = op_sub; 
    table->op_table[MF_OP_MUL] = op_mul; 
    table->op_table[MF_OP_DIV] = op_div;
    
    table->op_table[MF_OP_SIN] = op_sin; 
    table->op_table[MF_OP_COS] = op_cos; 
    table->op_table[MF_OP_FLOOR] = op_floor; 
    table->op_table[MF_OP_CEIL] = op_ceil;
    table->op_table[MF_OP_ABS] = op_abs; 
    table->op_table[MF_OP_SQRT] = op_sqrt; 
    table->op_table[MF_OP_ATAN2] = op_atan2; 
    table->op_table[MF_OP_POW] = op_pow;
    
    table->op_table[MF_OP_MIN] = op_min; 
    table->op_table[MF_OP_MAX] = op_max;
    
    table->op_table[MF_OP_SMOOTHSTEP] = op_smoothstep;
    table->op_table[MF_OP_STEP] = op_step;
}
