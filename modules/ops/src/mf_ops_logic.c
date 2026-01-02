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
MF_KERNEL_LOGIC(xor, !=)

MF_KERNEL_UNARY_GENERIC(not, u8, u8, U8, !v, u8, u8)

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
    
    dst->info.dtype = true_val->info.dtype; 
    if (!mf_utils_resolve_ternary_shape(ctx, dst, cond, true_val, false_val)) return;
    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t sz_dst = mf_tensor_count(dst);

    mf_accessor_f32 it_c = mf_accessor_f32_begin(cond); // Use F32 base for condition check if needed
    mf_accessor_f32 it_t = mf_accessor_f32_begin(true_val);
    mf_accessor_f32 it_f = mf_accessor_f32_begin(false_val);
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);

    bool cond_is_f32 = (cond->info.dtype == MF_DTYPE_F32);
    size_t esize = mf_dtype_size(dst->info.dtype);

    for(size_t i=0; i<sz_dst; ++i) {
        bool condition = cond_is_f32 ? (mf_accessor_f32_get(&it_c) != 0.0f) : (*((u8*)it_c.it.ptr) != 0);
        
        memcpy(it_dst.it.ptr, condition ? it_t.it.ptr : it_f.it.ptr, esize);

        mf_accessor_f32_advance(&it_c, inst->strides[1]);
        mf_accessor_f32_advance(&it_t, inst->strides[2]);
        mf_accessor_f32_advance(&it_f, inst->strides[3]);
        mf_accessor_f32_advance(&it_dst, inst->strides[0]);
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
    table[MF_OP_XOR] = op_xor;
    table[MF_OP_NOT] = op_not;
    table[MF_OP_SELECT] = op_select;
}
