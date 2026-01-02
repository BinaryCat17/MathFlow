#include "mf_ops_internal.h"
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mathflow/base/mf_log.h>

// ... (op_range, op_cumsum, _check_mask_ptr, op_compress remain the same)

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
    mf_accessor_f32 d = mf_accessor_f32_begin(dst);

    for (int i = 0; i < count; ++i) {
        mf_accessor_f32_set(&d, (f32)i);
        mf_accessor_f32_advance(&d, 1);
    }
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
    mf_accessor_f32 it_src = mf_accessor_f32_begin(src);
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);

    f32 sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += mf_accessor_f32_get(&it_src);
        mf_accessor_f32_set(&it_dst, sum);
        mf_accessor_f32_advance(&it_src, inst->strides[1]);
        mf_accessor_f32_advance(&it_dst, inst->strides[0]);
    }
}

static inline bool _check_mask(const mf_accessor_u8* it) {
    return mf_accessor_u8_get(it) != 0;
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
    mf_accessor_u8 it_count = mf_accessor_u8_begin(mask);
    for(size_t i=0; i<count; ++i) {
        if (_check_mask(&it_count)) true_count++;
        mf_accessor_u8_advance(&it_count, inst->strides[2]);
    }

    dst->info.dtype = data->info.dtype;
    int32_t new_shape[] = { (int32_t)true_count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, new_shape, 1)) return;

    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t write_idx = 0;
    size_t elem_size = mf_dtype_size(data->info.dtype);
    mf_accessor_f32 it_data = mf_accessor_f32_begin(data); // Assume F32 for now, but should be generic
    mf_accessor_u8 it_mask = mf_accessor_u8_begin(mask);
    u8* dst_raw = (u8*)mf_tensor_data(dst); 

    for(size_t i=0; i<count; ++i) {
        if (_check_mask(&it_mask)) {
            memcpy(dst_raw + write_idx * elem_size, it_data.it.ptr, elem_size);
            write_idx++;
        }
        mf_accessor_f32_advance(&it_data, inst->strides[1]);
        mf_accessor_u8_advance(&it_mask, inst->strides[2]);
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
    mf_accessor_f32 d = mf_accessor_f32_begin(dst);
    
    u32 axis_stride = 1;
    for (int i = ctx->ndim - 1; i > axis; --i) axis_stride *= ctx->domain_shape[i];
    u32 axis_size = ctx->domain_shape[axis];
    if (axis_size == 0) axis_size = 1;

    u32 start_linear = ctx->linear_offset;
    for (size_t i = 0; i < count; ++i) {
        u32 global_idx = start_linear + (u32)i;
        mf_accessor_f32_set(&d, (f32)((global_idx / axis_stride) % axis_size));
        mf_accessor_f32_advance(&d, inst->strides[0]);
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
    
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);
    mf_accessor_f32 it_idx = mf_accessor_f32_begin(src_indices);
    bool idx_is_f32 = (src_indices->info.dtype == MF_DTYPE_F32);

    for (size_t i = 0; i < out_count; ++i) {
        int idx = -1;
        if (idx_is_f32) {
            // Use safe get to handle NaN/Inf indices
            f32 f_idx = mf_accessor_f32_get_safe(&it_idx);
            idx = (int)f_idx;
        } else {
            // Re-map as i32 accessor for integers
            idx = *((int32_t*)it_idx.it.ptr);
        }

        if (idx >= 0 && (size_t)idx < data_count) {
            void* src_ptr = mf_tensor_iter_get_at_linear(src_data, (size_t)idx);
            memcpy(it_dst.it.ptr, src_ptr, elem_size);
        } else {
            // RELIABILITY: Safe default instead of crash
            memset(it_dst.it.ptr, 0, elem_size);
            
            if (_mf_should_log_error(ctx)) {
                ctx->error = MF_ERROR_OUT_OF_BOUNDS;
                ctx->error_idx = (u32)i;
                MF_LOG_ERROR("Gather OOB: Index %d at batch element %zu. Data size: %zu. Using 0.", 
                             idx, i, data_count);
            }
        }
        mf_accessor_f32_advance(&it_idx, inst->strides[2]);
        mf_accessor_f32_advance(&it_dst, inst->strides[0]);
    }
}

void mf_ops_array_register(mf_op_func* table) {
    table[MF_OP_RANGE] = op_range;
    table[MF_OP_INDEX] = op_index;
    table[MF_OP_GATHER] = op_gather;
    table[MF_OP_CUMSUM] = op_cumsum;
    table[MF_OP_COMPRESS] = op_compress;
}