#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>

static void op_copy(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    const mf_type_info* info = &ctx->reg_info[inst->src1_idx];
    size_t sz = ctx->batch_size;
    size_t esize = mf_dtype_size(info->dtype);
    
    u8* s_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];

    i32 st0 = MF_GET_STRIDE_D(inst) * (i32)esize;
    i32 st1 = MF_GET_STRIDE_S1(inst) * (i32)esize;

    for(size_t i=0; i<sz; ++i) {
        memcpy(d_ptr, s_ptr, esize);
        s_ptr += st1;
        d_ptr += st0;
    }
}

// Slice(Input, Range) -> View. Range is [Start, Count]
static void op_slice(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    // In our new architecture, metadata-only ops should be handled by dispatcher.
    // As a kernel, it falls back to a stride-aware copy.
    op_copy(ctx, inst);
}

// Reshape(Input, ShapeTensor) -> View
static void op_reshape(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    op_copy(ctx, inst);
}

void mf_ops_register_state(mf_op_func* table) {
    table[MF_OP_COPY] = op_copy;
    table[MF_OP_SLICE] = op_slice;
    table[MF_OP_RESHAPE] = op_reshape;
}
