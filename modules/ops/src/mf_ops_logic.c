#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include "mf_ops_internal.h"
#include <string.h>

typedef struct mf_tensor mf_tensor;

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
static void op_select(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    mf_tensor* dst = &ctx->registers[inst->dest_idx];
    mf_tensor* cond = &ctx->registers[inst->src1_idx];
    mf_tensor* true_val = &ctx->registers[inst->src2_idx];
    mf_tensor* false_val = &ctx->registers[inst->src3_idx];
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, cond);
    MF_CHECK_INPUT(ctx, true_val);
    MF_CHECK_INPUT(ctx, false_val);
    
    dst->info.dtype = true_val->info.dtype; 
    if (!mf_utils_resolve_ternary_shape(ctx, dst, cond, true_val, false_val)) return;
    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t sz_dst = mf_tensor_count(dst);
    size_t esize = mf_dtype_size(dst->info.dtype);

    mf_tensor_iter it_c = mf_tensor_iter_begin(cond);
    mf_tensor_iter it_t = mf_tensor_iter_begin(true_val);
    mf_tensor_iter_begin(false_val); // Wait, false_val was missing in the original code's iterator start? No, it's there below.
    mf_tensor_iter it_f = mf_tensor_iter_begin(false_val);
    mf_tensor_iter it_d = mf_tensor_iter_begin(dst);

    bool cond_is_f32 = (cond->info.dtype == MF_DTYPE_F32);

    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);
    i32 st2 = MF_GET_STRIDE_S2(inst);
    i32 st3 = MF_GET_STRIDE_S3(inst);

    for(size_t i=0; i<sz_dst; ++i) {
        bool condition = false;
        if (cond_is_f32) {
            condition = (*((f32*)it_c.ptr) != 0.0f);
        } else {
            condition = (*((u8*)it_c.ptr) != 0);
        }
        
        memcpy(it_d.ptr, condition ? it_t.ptr : it_f.ptr, esize);

        mf_tensor_iter_advance(&it_c, st1);
        mf_tensor_iter_advance(&it_t, st2);
        mf_tensor_iter_advance(&it_f, st3);
        mf_tensor_iter_advance(&it_d, st0);
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