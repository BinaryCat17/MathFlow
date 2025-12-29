#include <mathflow/ops/mf_ops_array.h>
#include <mathflow/ops/mf_kernel_utils.h>
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>

// --- Op: Range (Iota) ---
// Src1: Scalar count (e.g. 5)
// Dest: Vector [0, 1, 2, 3, 4]
static void op_range(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* count_tensor = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    if (!dst || !count_tensor) return;

    // Determine count
    int count = 0;
    if (count_tensor->dtype == MF_DTYPE_F32) {
        count = (int)((f32*)count_tensor->data)[0];
    } else if (count_tensor->dtype == MF_DTYPE_I32) {
        count = ((int32_t*)count_tensor->data)[0];
    }
    
    if (count < 0) count = 0;

    // Resize Dest
    dst->dtype = MF_DTYPE_F32; // Always F32 for now
    int32_t shape[] = { count };
    if (!ctx->resize_tensor(ctx->impl, dst, shape, 1)) return;

    // Fill
    f32* d = (f32*)dst->data;
    for (int i = 0; i < count; ++i) {
        d[i] = (f32)i;
    }
}

// --- Op: CumSum (Prefix Sum) ---
// Src1: Vector [10, 20, 30]
// Dest: Vector [10, 30, 60]
static void op_cumsum(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    if (!dst || !src) return;

    // Shape matches source
    if (!mf_utils_resolve_unary_shape(ctx, dst, src)) return;

    // Implementation (F32 only for now)
    if (src->dtype != MF_DTYPE_F32) return; // TODO: Support others

    f32* s = (f32*)src->data;
    f32* d = (f32*)dst->data;
    f32 sum = 0.0f;
    for (size_t i = 0; i < dst->size; ++i) {
        sum += s[i];
        d[i] = sum;
    }
}

// --- Op: Compress (Filter) ---
// Src1: Data [10, 20, 30]
// Src2: Mask [1, 0, 1]
// Dest: [10, 30]
static void op_compress(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* data = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    mf_tensor* mask = ctx->map_tensor(ctx->impl, src2_idx, MF_ACCESS_READ);
    
    if (!dst || !data || !mask) return;

    // Validate size (Must match)
    // TODO: Broadcasting logic? For compress usually 1-to-1.
    size_t count = (data->size < mask->size) ? data->size : mask->size;
    
    // Pass 1: Count True
    size_t true_count = 0;
    for(size_t i=0; i<count; ++i) {
        bool keep = false;
        if (mask->dtype == MF_DTYPE_U8) keep = ((u8*)mask->data)[i] != 0;
        else if (mask->dtype == MF_DTYPE_F32) keep = ((f32*)mask->data)[i] > 0.5f; // Threshold
        else if (mask->dtype == MF_DTYPE_I32) keep = ((int32_t*)mask->data)[i] != 0;
        
        if (keep) true_count++;
    }

    // Resize Dest
    dst->dtype = data->dtype;
    int32_t new_shape[] = { (int32_t)true_count };
    if (!ctx->resize_tensor(ctx->impl, dst, new_shape, 1)) return;

    // Pass 2: Copy
    size_t write_idx = 0;
    size_t elem_size = mf_dtype_size(data->dtype);
    u8* src_ptr = (u8*)data->data;
    u8* dst_ptr = (u8*)dst->data;

    for(size_t i=0; i<count; ++i) {
        bool keep = false;
        if (mask->dtype == MF_DTYPE_U8) keep = ((u8*)mask->data)[i] != 0;
        else if (mask->dtype == MF_DTYPE_F32) keep = ((f32*)mask->data)[i] > 0.5f;
        else if (mask->dtype == MF_DTYPE_I32) keep = ((int32_t*)mask->data)[i] != 0;

        if (keep) {
            memcpy(dst_ptr + write_idx * elem_size, src_ptr + i * elem_size, elem_size);
            write_idx++;
        }
    }
}

// --- Op: Index (Intrinsic Coordinate) ---
// Src1: Axis (0=Y, 1=X, etc)
// Dest: Vector of global coordinates
static void op_index(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* axis_t = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    if (!dst || !axis_t) return;

    int axis = 0;
    if (axis_t->dtype == MF_DTYPE_F32) axis = (int)((f32*)axis_t->data)[0];
    else if (axis_t->dtype == MF_DTYPE_I32) axis = ((int32_t*)axis_t->data)[0];
    
    // Determine Output Size
    // If batch_size is set, use it. Otherwise 1 (Scalar)? 
    // Usually OP_INDEX implies we are in a parallel context.
    size_t count = (ctx->batch_size > 0) ? ctx->batch_size : 1;

    dst->dtype = MF_DTYPE_F32;
    int32_t shape[] = { (int32_t)count };
    if (!ctx->resize_tensor(ctx->impl, dst, shape, 1)) return;

    f32* d = (f32*)dst->data;
    
    // Generation Logic (Assumes 2D Tile Layout: Row-Major)
    // Axis 1 is Fastest (X), Axis 0 is Slowest (Y)
    u32 width = ctx->local_size[1];
    if (width == 0) width = 1; // Safety
    
    u32 offset = ctx->global_offset[axis];
    
    for (size_t i = 0; i < count; ++i) {
        u32 coord = 0;
        if (axis == 1) {
            // X: Modulo Width
            coord = offset + (i % width);
        } else if (axis == 0) {
            // Y: Divide Width
            coord = offset + (i / width);
        } else {
            // Z or others: For now assume 0 or direct mapping if 1D
             coord = offset;
        }
        d[i] = (f32)coord;
    }
}

// --- Op: Resolution (Domain Size) ---
// Src1: Axis (0=Height, 1=Width)
// Dest: Scalar Size
static void op_resolution(const mf_kernel_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = ctx->map_tensor(ctx->impl, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* axis_t = ctx->map_tensor(ctx->impl, src1_idx, MF_ACCESS_READ);
    if (!dst || !axis_t) return;

    int axis = 0;
    if (axis_t->dtype == MF_DTYPE_F32) axis = (int)((f32*)axis_t->data)[0];
    else if (axis_t->dtype == MF_DTYPE_I32) axis = ((int32_t*)axis_t->data)[0];
    
    // Output: Scalar [1]
    dst->dtype = MF_DTYPE_F32;
    int32_t shape[] = { 1 };
    if (!ctx->resize_tensor(ctx->impl, dst, shape, 0)) return; // 0 dimensions = scalar

    f32* d = (f32*)dst->data;
    if (axis >= 0 && axis < 3) {
        d[0] = (f32)ctx->global_size[axis];
    } else {
        d[0] = 0.0f;
    }
}

// --- Registration ---
void mf_ops_array_register(mf_backend_dispatch_table* table) {
    table->op_table[MF_OP_RANGE] = op_range;
    table->op_table[MF_OP_INDEX] = op_index;
    table->op_table[MF_OP_RESOLUTION] = op_resolution;
    table->op_table[MF_OP_CUMSUM] = op_cumsum;
    table->op_table[MF_OP_COMPRESS] = op_compress;
}