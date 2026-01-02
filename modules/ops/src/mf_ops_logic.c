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

MF_KERNEL_UNARY_GENERIC(not, u8, u8, U8, !v)

// --- Selection ---
static void op_select(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    size_t sz = ctx->batch_size;
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    u8*  c_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    f32* t_ptr = (f32*)ctx->reg_ptrs[inst->src2_idx];
    f32* f_ptr = (f32*)ctx->reg_ptrs[inst->src3_idx];

    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);
    i32 st2 = MF_GET_STRIDE_S2(inst);
    i32 st3 = MF_GET_STRIDE_S3(inst);

    for (size_t i = 0; i < sz; ++i) {
        *d_ptr = (*c_ptr) ? (*t_ptr) : (*f_ptr);
        
        c_ptr += st1;
        t_ptr += st2;
        f_ptr += st3;
        d_ptr += st0;
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