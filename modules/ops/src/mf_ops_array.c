#include "mf_ops_internal.h"
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>
#include <stdio.h>
#include <mathflow/base/mf_log.h>

// --- Op: Range (Iota) ---
static void op_range(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* count_tensor = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, count_tensor);

    int count = mf_utils_get_scalar_int(count_tensor);
    if (count < 0) count = 0;

    dst->info.dtype = MF_DTYPE_F32;
    int32_t shape[] = { count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, shape, 1)) return;

    MF_CHECK_DST_DATA(ctx, dst);
    f32* d = (f32*)mf_tensor_data(dst);

    for (int i = 0; i < count; ++i) d[i] = (f32)i;
}

// --- Op: CumSum (Prefix Sum) ---
static void op_cumsum(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* src = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, src);

    if (!mf_utils_resolve_unary_shape(ctx, dst, src)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    if (src->info.dtype != MF_DTYPE_F32) return;

    size_t count = mf_tensor_count(dst);
    mf_tensor_iter it_src = mf_tensor_iter_begin(src);
    mf_tensor_iter it_dst = mf_tensor_iter_begin(dst);

    f32 sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += *((f32*)it_src.ptr);
        *((f32*)it_dst.ptr) = sum;
        mf_tensor_iter_advance(&it_src, inst->strides[1]);
        mf_tensor_iter_advance(&it_dst, inst->strides[0]);
    }
}

static inline bool _check_mask_ptr(const mf_tensor* mask, void* ptr) {
    if (mask->info.dtype == MF_DTYPE_U8) return *((u8*)ptr) != 0;
    if (mask->info.dtype == MF_DTYPE_F32) return *((f32*)ptr) > 0.5f;
    if (mask->info.dtype == MF_DTYPE_I32) return *((int32_t*)ptr) != 0;
    return false;
}

// --- Op: Compress (Filter) ---
static void op_compress(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* data = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* mask = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, data);
    MF_CHECK_INPUT(ctx, mask);

    size_t count_data = mf_tensor_count(data);
    size_t count_mask = mf_tensor_count(mask);
    size_t count = (count_data < count_mask) ? count_data : count_mask;
    
    size_t true_count = 0;
    mf_tensor_iter it_count = mf_tensor_iter_begin(mask);
    for(size_t i=0; i<count; ++i) {
        if (_check_mask_ptr(mask, it_count.ptr)) true_count++;
        mf_tensor_iter_advance(&it_count, inst->strides[2]);
    }

    dst->info.dtype = data->info.dtype;
    int32_t new_shape[] = { (int32_t)true_count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, new_shape, 1)) return;

    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t write_idx = 0;
    size_t elem_size = mf_dtype_size(data->info.dtype);
    mf_tensor_iter it_data = mf_tensor_iter_begin(data);
    mf_tensor_iter it_mask = mf_tensor_iter_begin(mask);
    u8* dst_ptr = (u8*)mf_tensor_data(dst); // Dst is always contiguous after resize

    for(size_t i=0; i<count; ++i) {
        if (_check_mask_ptr(mask, it_mask.ptr)) {
            memcpy(dst_ptr + write_idx * elem_size, it_data.ptr, elem_size);
            write_idx++;
        }
        mf_tensor_iter_advance(&it_data, inst->strides[1]);
        mf_tensor_iter_advance(&it_mask, inst->strides[2]);
    }
}

// --- Op: Index (Intrinsic Coordinate) ---
static void op_index(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* axis_t = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, axis_t);

    int axis = mf_utils_get_scalar_int(axis_t);
    if (axis < 0 || axis >= MF_MAX_DIMS) axis = 0;

    size_t count = (ctx->batch_size > 0) ? ctx->batch_size : 1;

    dst->info.dtype = MF_DTYPE_F32;
    int32_t shape[] = { (int32_t)count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, shape, 1)) return;

    MF_CHECK_DST_DATA(ctx, dst);
    f32* d = (f32*)mf_tensor_data(dst);
    
    u32 axis_stride = 1;
    for (int i = ctx->ndim - 1; i > axis; --i) axis_stride *= ctx->domain_shape[i];
    u32 axis_size = ctx->domain_shape[axis];
    if (axis_size == 0) axis_size = 1;

    u32 start_linear = ctx->linear_offset;
    for (size_t i = 0; i < count; ++i) {
        u32 global_idx = start_linear + (u32)i;
        d[i] = (f32)((global_idx / axis_stride) % axis_size);
    }
}

// --- Op: Gather (Random Access) ---
static void op_gather(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* src_data = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* src_indices = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, src_data);
    MF_CHECK_INPUT(ctx, src_indices);

    if (!mf_utils_resolve_unary_shape(ctx, dst, src_indices)) return;
    dst->info.dtype = src_data->info.dtype;

    MF_CHECK_DST_DATA(ctx, dst);

    size_t data_count = mf_tensor_count(src_data);
    size_t out_count = mf_tensor_count(dst);
    size_t elem_size = mf_dtype_size(src_data->info.dtype);
    
    mf_tensor_iter it_dst = mf_tensor_iter_begin(dst);
    mf_tensor_iter it_idx = mf_tensor_iter_begin(src_indices);
    bool idx_is_f32 = (src_indices->info.dtype == MF_DTYPE_F32);

    for (size_t i = 0; i < out_count; ++i) {
        int idx = idx_is_f32 ? (int)(*((f32*)it_idx.ptr)) : (*((int32_t*)it_idx.ptr));
        if (idx >= 0 && (size_t)idx < data_count) {
            void* src_ptr = mf_tensor_iter_get_at_linear(src_data, (size_t)idx);
            memcpy(it_dst.ptr, src_ptr, elem_size);
        } else {
            MF_LOG_ERROR("Gather out of bounds! Index %d (Data Size: %zu)", idx, data_count);
            ctx->error = MF_ERROR_OUT_OF_BOUNDS;
            break;
        }
        mf_tensor_iter_next(&it_idx);
        mf_tensor_iter_next(&it_dst);
    }
}

void mf_ops_array_register(mf_op_func* table) {
    table[MF_OP_RANGE] = op_range;
    table[MF_OP_INDEX] = op_index;
    table[MF_OP_GATHER] = op_gather;
    table[MF_OP_CUMSUM] = op_cumsum;
    table[MF_OP_COMPRESS] = op_compress;
}