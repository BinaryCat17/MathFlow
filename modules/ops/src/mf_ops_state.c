#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/ops/mf_kernel_utils.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>

static void op_copy(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    if (!dst || !src) return;

    dst->dtype = src->dtype;
    if (!ctx->resize_tensor(ctx->impl, dst, src->shape, src->ndim)) return;

    size_t size = mf_dtype_size(src->dtype) * src->size;
    memcpy(dst->data, src->data, size);
}

void mf_ops_register_state(mf_backend_dispatch_table* table) {
    table->op_table[MF_OP_COPY] = op_copy;
}
