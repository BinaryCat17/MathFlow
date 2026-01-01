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

    f32* s = (f32*)mf_tensor_data(src);
    f32* d = (f32*)mf_tensor_data(dst);
    f32 sum = 0.0f;
    size_t count = mf_tensor_count(dst);
    for (size_t i = 0; i < count; ++i) {
        sum += s[i];
        d[i] = sum;
    }
}

static inline bool _check_mask(const mf_tensor* mask, size_t i) {
    void* d = mf_tensor_data(mask);
    if (mask->info.dtype == MF_DTYPE_U8) return ((u8*)d)[i] != 0;
    if (mask->info.dtype == MF_DTYPE_F32) return ((f32*)d)[i] > 0.5f;
    if (mask->info.dtype == MF_DTYPE_I32) return ((int32_t*)d)[i] != 0;
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
    for(size_t i=0; i<count; ++i) if (_check_mask(mask, i)) true_count++;

    dst->info.dtype = data->info.dtype;
    int32_t new_shape[] = { (int32_t)true_count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, new_shape, 1)) return;

    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t write_idx = 0;
    size_t elem_size = mf_dtype_size(data->info.dtype);
    u8* src_ptr = (u8*)mf_tensor_data(data);
    u8* dst_ptr = (u8*)mf_tensor_data(dst);

    for(size_t i=0; i<count; ++i) {
        if (_check_mask(mask, i)) {
            memcpy(dst_ptr + write_idx * elem_size, src_ptr + i * elem_size, elem_size);
            write_idx++;
        }
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

    u32 start_linear = ctx->tile_offset[0];
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
    
    u8* data_ptr = (u8*)mf_tensor_data(src_data);
    u8* out_ptr = (u8*)mf_tensor_data(dst);
    void* idx_ptr = mf_tensor_data(src_indices);
    bool idx_is_f32 = (src_indices->info.dtype == MF_DTYPE_F32);

    for (size_t i = 0; i < out_count; ++i) {
        int idx = idx_is_f32 ? (int)((f32*)idx_ptr)[i] : ((int32_t*)idx_ptr)[i];
        if (idx >= 0 && (size_t)idx < data_count) {
            memcpy(out_ptr + i * elem_size, data_ptr + idx * elem_size, elem_size);
        } else {
            memset(out_ptr + i * elem_size, 0, elem_size);
            ctx->error = MF_ERROR_OUT_OF_BOUNDS;
        }
    }
}

void mf_ops_array_register(mf_op_func* table) {
    table[MF_OP_RANGE] = op_range;
    table[MF_OP_INDEX] = op_index;
    table[MF_OP_GATHER] = op_gather;
    table[MF_OP_CUMSUM] = op_cumsum;
    table[MF_OP_COMPRESS] = op_compress;
}