#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>

typedef struct mf_tensor mf_tensor;

static void op_copy(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    mf_tensor* dst = &ctx->registers[inst->dest_idx];
    mf_tensor* src = &ctx->registers[inst->src1_idx];
    if (!dst || !src) return;

    bool dst_allocated = (dst->buffer != NULL);
    bool broadcast = mf_tensor_is_scalar(src) && !mf_tensor_is_scalar(dst) && dst_allocated;

    if (broadcast) {
        size_t count = mf_tensor_count(dst);
        void* d_ptr = mf_tensor_data(dst);
        
        if (dst->info.dtype == MF_DTYPE_F32) {
             f32 val = mf_utils_get_scalar_f32(src);
             f32* d = (f32*)d_ptr;
             for(size_t i=0; i<count; ++i) d[i] = val;
        } else if (dst->info.dtype == MF_DTYPE_I32) {
             i32 val = mf_utils_get_scalar_int(src);
             i32* d = (i32*)d_ptr;
             for(size_t i=0; i<count; ++i) d[i] = val;
        } else if (dst->info.dtype == MF_DTYPE_U8) {
             u8 val = (u8)mf_utils_get_scalar_int(src);
             u8* d = (u8*)d_ptr;
             for(size_t i=0; i<count; ++i) d[i] = val;
        }
        return;
    }

    dst->info.dtype = src->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, src->info.shape, src->info.ndim)) return;
    MF_CHECK_DST_DATA(ctx, dst);
    MF_CHECK_INPUT(ctx, src);

    size_t count = mf_tensor_count(src);
    size_t elem_size = mf_dtype_size(src->info.dtype);
    
    mf_tensor_iter it_src = mf_tensor_iter_begin(src);
    mf_tensor_iter it_dst = mf_tensor_iter_begin(dst);

    for(size_t i=0; i<count; ++i) {
        memcpy(it_dst.ptr, it_src.ptr, elem_size);
        mf_tensor_iter_next(&it_src);
        mf_tensor_iter_next(&it_dst);
    }
}

// Slice(Input, Range) -> View. Range is [Start, Count]
static void op_slice(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    mf_tensor* dst = &ctx->registers[inst->dest_idx];
    mf_tensor* src = &ctx->registers[inst->src1_idx];
    mf_tensor* range = &ctx->registers[inst->src2_idx];
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, src);
    MF_CHECK_INPUT(ctx, range);
    
    void* r_ptr = mf_tensor_data(range);
    int start = (range->info.dtype == MF_DTYPE_F32) ? (int)((f32*)r_ptr)[0] : ((int32_t*)r_ptr)[0];
    int count = (range->info.dtype == MF_DTYPE_F32) ? (int)((f32*)r_ptr)[1] : ((int32_t*)r_ptr)[1];
    
    if (start < 0) start = 0;
    if (count < 0) count = 0;
    
    mf_tensor_slice(dst, src, (size_t)start, (size_t)count);
}

// Reshape(Input, ShapeTensor) -> View
static void op_reshape(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    mf_tensor* dst = &ctx->registers[inst->dest_idx];
    mf_tensor* src = &ctx->registers[inst->src1_idx];
    mf_tensor* shape_t = &ctx->registers[inst->src2_idx];
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, src);
    MF_CHECK_INPUT(ctx, shape_t);
    
    int32_t new_shape[MF_MAX_DIMS];
    size_t ndim = mf_tensor_count(shape_t);
    if (ndim > MF_MAX_DIMS) ndim = MF_MAX_DIMS;
    
    void* s_ptr = mf_tensor_data(shape_t);
    bool is_f32 = (shape_t->info.dtype == MF_DTYPE_F32);
    for(int i=0; i<(int)ndim; ++i) {
        new_shape[i] = is_f32 ? (int32_t)((f32*)s_ptr)[i] : ((int32_t*)s_ptr)[i];
    }
    
    mf_tensor_reshape(dst, src, new_shape, (int)ndim);
}

void mf_ops_register_state(mf_op_func* table) {
    table[MF_OP_COPY] = op_copy;
    table[MF_OP_SLICE] = op_slice;
    table[MF_OP_RESHAPE] = op_reshape;
}
