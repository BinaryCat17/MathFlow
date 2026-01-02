#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include "mf_ops_internal.h"
#include <string.h>

// --- Comparison ---
MF_KERNEL_COMPARE(LESS, <)
MF_KERNEL_COMPARE(GREATER, >)
MF_KERNEL_COMPARE(EQUAL, ==)
MF_KERNEL_COMPARE(NEQUAL, !=)
MF_KERNEL_COMPARE(LEQUAL, <=)
MF_KERNEL_COMPARE(GEQUAL, >=)

// --- Logic ---
MF_KERNEL_LOGIC(AND, &&)
MF_KERNEL_LOGIC(OR, ||)
MF_KERNEL_LOGIC(XOR, !=)

MF_KERNEL_UNARY(NOT, u8, u8, !v)

// --- Selection ---
void op_SELECT(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
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
