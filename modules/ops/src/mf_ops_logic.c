#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include "mf_ops_internal.h"
#include <string.h>

// --- Comparison ---
MF_KERNEL_COMPARE(less, <)
MF_KERNEL_COMPARE(greater, >)
MF_KERNEL_COMPARE(equal, ==)
MF_KERNEL_COMPARE(nequal, !=)
MF_KERNEL_COMPARE(lequal, <=)
MF_KERNEL_COMPARE(gequal, >=)

// --- Logic ---
MF_KERNEL_LOGIC(and, &&)
MF_KERNEL_LOGIC(or, ||)

MF_KERNEL_UNARY_GENERIC(not, u8, u8, U8, !v)

// --- Selection ---
static void op_select(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* true_val = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    mf_tensor* false_val = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, cond);
    MF_CHECK_INPUT(ctx, true_val);
    MF_CHECK_INPUT(ctx, false_val);
    
    // Enforce type from True Branch
    dst->info.dtype = true_val->info.dtype; 
    if (!mf_utils_resolve_ternary_shape(ctx, dst, cond, true_val, false_val)) return;
    MF_CHECK_DST_DATA(ctx, dst);
    
    u8* c = (u8*)mf_tensor_data(cond);
    f32* c_f32 = (f32*)c;
    bool cond_is_f32 = (cond->info.dtype == MF_DTYPE_F32);

    f32* dt = (f32*)mf_tensor_data(true_val);
    f32* df = (f32*)mf_tensor_data(false_val);
    f32* dd = (f32*)mf_tensor_data(dst);

    size_t sz_c = mf_tensor_count(cond);
    size_t sz_t = mf_tensor_count(true_val);
    size_t sz_f = mf_tensor_count(false_val);
    size_t sz_dst = mf_tensor_count(dst);

    for(size_t i=0; i<sz_dst; ++i) {
        bool condition = cond_is_f32 ? (c_f32[i % sz_c] != 0.0f) : (c[i % sz_c] != 0);
        dd[i] = condition ? dt[i % sz_t] : df[i % sz_f];
    }
}

void mf_ops_register_logic(mf_op_func* table) {
    table[MF_OP_LESS] = op_less;
    table[MF_OP_GREATER] = op_greater;
    table[MF_OP_EQUAL] = op_equal;
    table[MF_OP_NEQUAL] = op_nequal;
    table[MF_OP_LEQUAL] = op_lequal;
    table[MF_OP_GEQUAL] = op_gequal;
    table[MF_OP_AND] = op_and;
    table[MF_OP_OR] = op_or;
    table[MF_OP_NOT] = op_not;
    table[MF_OP_SELECT] = op_select;
}
