#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_math.h>
#include "mf_ops_internal.h"
#include <string.h>

static void op_copy(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !src) return;

    dst->info.dtype = src->info.dtype;
    if (!mf_exec_ctx_resize_tensor(ctx, dst, src->info.shape, src->info.ndim)) return;

    size_t size = mf_tensor_size_bytes(src);
    memcpy(mf_tensor_data(dst), mf_tensor_data(src), size);
}

// Slice(Input, Range) -> View
// Range is a Vec2: [Start, Count]
static void op_slice(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* range = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    
    if (!dst || !src || !range) return;
    
    int start = 0;
    int count = 0;
    
    size_t range_size = mf_tensor_count(range);
    if (range_size >= 2) {
        if (range->info.dtype == MF_DTYPE_F32) {
            start = (int)((f32*)mf_tensor_data(range))[0];
            count = (int)((f32*)mf_tensor_data(range))[1];
        } else if (range->info.dtype == MF_DTYPE_I32) {
            start = (int)((i32*)mf_tensor_data(range))[0];
            count = (int)((i32*)mf_tensor_data(range))[1];
        }
    } else {
         return; 
    }
    
    if (start < 0) start = 0;
    if (count < 0) count = 0;
    
    mf_tensor_slice(dst, src, (size_t)start, (size_t)count);
}

// Reshape(Input, ShapeTensor) -> View
static void op_reshape(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* shape_t = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    
    if (!dst || !src || !shape_t) return;
    
    int32_t new_shape[MF_MAX_DIMS];
    size_t shape_len = mf_tensor_count(shape_t);
    int ndim = (int)shape_len;
    if (ndim > MF_MAX_DIMS) ndim = MF_MAX_DIMS;
    
    for(int i=0; i<ndim; ++i) {
        if (shape_t->info.dtype == MF_DTYPE_F32) {
            new_shape[i] = (int32_t)((f32*)mf_tensor_data(shape_t))[i];
        } else if (shape_t->info.dtype == MF_DTYPE_I32) {
            new_shape[i] = ((int32_t*)mf_tensor_data(shape_t))[i];
        }
    }
    
    mf_tensor_reshape(dst, src, new_shape, ndim);
}

void mf_ops_register_state(mf_op_func* table) {
    table[MF_OP_COPY] = op_copy;
    table[MF_OP_SLICE] = op_slice;
    table[MF_OP_RESHAPE] = op_reshape;
}
