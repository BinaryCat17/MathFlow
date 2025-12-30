#include "mf_ops_internal.h"
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>

// --- Op: Range (Iota) ---
// Src1: Scalar count (e.g. 5)
// Dest: Vector [0, 1, 2, 3, 4]
static void op_range(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* count_tensor = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !count_tensor) return;

    // Determine count
    int count = 0;
    if (count_tensor->info.dtype == MF_DTYPE_F32) {
        count = (int)((f32*)mf_tensor_data(count_tensor))[0];
    } else if (count_tensor->info.dtype == MF_DTYPE_I32) {
        count = ((int32_t*)mf_tensor_data(count_tensor))[0];
    }
    
    if (count < 0) count = 0;

    // Resize Dest
    dst->info.dtype = MF_DTYPE_F32; // Always F32 for now
    int32_t shape[] = { count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, shape, 1)) return;

    // Fill
    f32* d = (f32*)mf_tensor_data(dst);
    for (int i = 0; i < count; ++i) {
        d[i] = (f32)i;
    }
}

// --- Op: CumSum (Prefix Sum) ---
// Src1: Vector [10, 20, 30]
// Dest: Vector [10, 30, 60]
static void op_cumsum(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !src) return;

    // Shape matches source
    if (!mf_utils_resolve_unary_shape(ctx, dst, src)) return;

    // Implementation (F32 only for now)
    if (src->info.dtype != MF_DTYPE_F32) return; // TODO: Support others

    f32* s = (f32*)mf_tensor_data(src);
    f32* d = (f32*)mf_tensor_data(dst);
    f32 sum = 0.0f;
    size_t count = mf_tensor_count(dst);
    for (size_t i = 0; i < count; ++i) {
        sum += s[i];
        d[i] = sum;
    }
}

// --- Op: Compress (Filter) ---
// Src1: Data [10, 20, 30]
// Src2: Mask [1, 0, 1]
// Dest: [10, 30]
static void op_compress(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* data = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* mask = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    
    if (!dst || !data || !mask) return;

    size_t data_count = mf_tensor_count(data);
    size_t mask_count = mf_tensor_count(mask);
    size_t count = (data_count < mask_count) ? data_count : mask_count;
    
    // Pass 1: Count True
    size_t true_count = 0;
    for(size_t i=0; i<count; ++i) {
        bool keep = false;
        if (mask->info.dtype == MF_DTYPE_U8) keep = ((u8*)mf_tensor_data(mask))[i] != 0;
        else if (mask->info.dtype == MF_DTYPE_F32) keep = ((f32*)mf_tensor_data(mask))[i] > 0.5f; // Threshold
        else if (mask->info.dtype == MF_DTYPE_I32) keep = ((int32_t*)mf_tensor_data(mask))[i] != 0;
        
        if (keep) true_count++;
    }

    // Resize Dest
    dst->info.dtype = data->info.dtype;
    int32_t new_shape[] = { (int32_t)true_count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, new_shape, 1)) return;

    // Pass 2: Copy
    size_t write_idx = 0;
    size_t elem_size = mf_dtype_size(data->info.dtype);
    u8* src_ptr = (u8*)mf_tensor_data(data);
    u8* dst_ptr = (u8*)mf_tensor_data(dst);

    for(size_t i=0; i<count; ++i) {
        bool keep = false;
        if (mask->info.dtype == MF_DTYPE_U8) keep = ((u8*)mf_tensor_data(mask))[i] != 0;
        else if (mask->info.dtype == MF_DTYPE_F32) keep = ((f32*)mf_tensor_data(mask))[i] > 0.5f;
        else if (mask->info.dtype == MF_DTYPE_I32) keep = ((int32_t*)mf_tensor_data(mask))[i] != 0;

        if (keep) {
            memcpy(dst_ptr + write_idx * elem_size, src_ptr + i * elem_size, elem_size);
            write_idx++;
        }
    }
}

// --- Op: Index (Intrinsic Coordinate) ---
// Src1: Axis (0=Slowest, e.g. Y/Batch, N=Fastest, e.g. X)
// Dest: Vector of global coordinates
static void op_index(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* axis_t = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !axis_t) return;

    int axis = 0;
    if (axis_t->info.dtype == MF_DTYPE_F32) axis = (int)((f32*)mf_tensor_data(axis_t))[0];
    else if (axis_t->info.dtype == MF_DTYPE_I32) axis = ((int32_t*)mf_tensor_data(axis_t))[0];
    
    // Safety check for axis
    if (axis < 0 || axis >= MF_MAX_DIMS) axis = 0;

    size_t count = (ctx->batch_size > 0) ? ctx->batch_size : 1;

    dst->info.dtype = MF_DTYPE_F32;
    int32_t shape[] = { (int32_t)count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, shape, 1)) return;

    f32* d = (f32*)mf_tensor_data(dst);
    
    // Pre-calculate strides for unflattening the batch index inside the tile
    // The batch corresponds to the TILE shape (ctx->tile_size).
    // We assume the tile execution is iterating over the tile dimensions in standard order.
    
    u32 tile_strides[MF_MAX_DIMS];
    u32 current_stride = 1;
    // Iterate from fastest dim (last) to slowest (0)
    for (int i = ctx->ndim - 1; i >= 0; --i) {
        tile_strides[i] = current_stride;
        current_stride *= ctx->tile_size[i];
    }
    
    u32 axis_offset = ctx->tile_offset[axis];
    u32 axis_stride = tile_strides[axis];
    
    // Optimization: If stride is 1 (Fastest Axis), simple increment
    if (axis_stride == 1) {
        for (size_t i = 0; i < count; ++i) {
            d[i] = (f32)(axis_offset + i);
        }
    } else {
        // Generic Unflattening (only for the requested axis)
        // local_coord = (i / stride) % size
        // But since we only need ONE axis, we can optimize.
        // Actually, "stride" here accumulates lower dims.
        // The formula is: (i / stride) % dim_size
        u32 dim_size = ctx->tile_size[axis];
        
        for (size_t i = 0; i < count; ++i) {
            u32 local_coord = (i / axis_stride) % dim_size;
            d[i] = (f32)(axis_offset + local_coord);
        }
    }
}

// --- Op: Resolution (Domain Size) ---
// Src1: Axis (0=Height, 1=Width)
// Dest: Scalar Size
static void op_resolution(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* axis_t = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !axis_t) return;

    int axis = 0;
    if (axis_t->info.dtype == MF_DTYPE_F32) axis = (int)((f32*)mf_tensor_data(axis_t))[0];
    else if (axis_t->info.dtype == MF_DTYPE_I32) axis = ((int32_t*)mf_tensor_data(axis_t))[0];
    
    // Output: Scalar [1]
    dst->info.dtype = MF_DTYPE_F32;
    int32_t shape[] = { 1 };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, shape, 0)) return; 

    f32* d = (f32*)mf_tensor_data(dst);
    if (axis >= 0 && axis < MF_MAX_DIMS) {
        d[0] = (f32)ctx->domain_shape[axis];
    } else {
        d[0] = 0.0f;
    }
}

// --- Registration ---
void mf_ops_array_register(mf_op_func* table) {
    table[MF_OP_RANGE] = op_range;
    table[MF_OP_INDEX] = op_index;
    table[MF_OP_RESOLUTION] = op_resolution;
    table[MF_OP_CUMSUM] = op_cumsum;
    table[MF_OP_COMPRESS] = op_compress;
}
