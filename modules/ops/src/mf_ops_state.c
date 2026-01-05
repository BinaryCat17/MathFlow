#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>

void op_COPY(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* info = &ctx->reg_info[inst->src1_idx];
    size_t sz = ctx->batch_size;
    size_t esize = mf_dtype_size(info->dtype);
    
    u8* s_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];

    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);

    for(size_t i=0; i<sz; ++i) {
        memcpy(d_ptr, s_ptr, esize);
        s_ptr += st1;
        d_ptr += st0;
    }
}

void op_SLICE(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    op_COPY(ctx, inst);
}

void op_RESHAPE(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    op_COPY(ctx, inst);
}