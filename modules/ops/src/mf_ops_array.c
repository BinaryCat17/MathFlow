#include "mf_ops_internal.h"
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mathflow/base/mf_log.h>

typedef struct mf_tensor mf_tensor;

// --- Op: CumSum (Prefix Sum) ---
static void op_cumsum(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    mf_tensor* dst = &ctx->registers[inst->dest_idx];
    mf_tensor* src = &ctx->registers[inst->src1_idx];
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, src);

    if (!mf_utils_resolve_unary_shape(ctx, dst, src)) return;
    MF_CHECK_DST_DATA(ctx, dst);

    if (src->info.dtype != MF_DTYPE_F32) return;

    size_t count = mf_tensor_count(dst);
    mf_accessor_f32 it_src = mf_accessor_f32_begin(src);
    mf_accessor_f32 it_dst = mf_accessor_f32_begin(dst);

    i32 st0 = MF_GET_STRIDE_D(inst);
    i32 st1 = MF_GET_STRIDE_S1(inst);

    f32 sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += mf_accessor_f32_get(&it_src);
        mf_accessor_f32_set(&it_dst, sum);
        mf_accessor_f32_advance(&it_src, st1);
        mf_accessor_f32_advance(&it_dst, st0);
    }
}

static inline bool _check_mask(const mf_accessor_u8* it) {
    return mf_accessor_u8_get(it) != 0;
}

// --- Op: Compress (Filter) ---
static void op_compress(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    mf_tensor* dst = &ctx->registers[inst->dest_idx];
    mf_tensor* data = &ctx->registers[inst->src1_idx];
    mf_tensor* mask = &ctx->registers[inst->src2_idx];
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, data);
    MF_CHECK_INPUT(ctx, mask);

    size_t count_data = mf_tensor_count(data);
    size_t count_mask = mf_tensor_count(mask);
    size_t count = (count_data < count_mask) ? count_data : count_mask;
    
    size_t true_count = 0;
    mf_accessor_u8 it_count = mf_accessor_u8_begin(mask);
    i32 st2 = MF_GET_STRIDE_S2(inst);

    for(size_t i=0; i<count; ++i) {
        if (_check_mask(&it_count)) true_count++;
        mf_accessor_u8_advance(&it_count, st2);
    }

    dst->info.dtype = data->info.dtype;
    int32_t new_shape[] = { (int32_t)true_count };
    if (!mf_exec_ctx_resize_tensor(ctx, dst, new_shape, 1)) return;

    MF_CHECK_DST_DATA(ctx, dst);
    
    size_t write_idx = 0;
    size_t elem_size = mf_dtype_size(data->info.dtype);
    mf_accessor_f32 it_data = mf_accessor_f32_begin(data); 
    mf_accessor_u8 it_mask = mf_accessor_u8_begin(mask);
    u8* dst_raw = (u8*)mf_tensor_data(dst); 

    i32 st1 = MF_GET_STRIDE_S1(inst);

    for(size_t i=0; i<count; ++i) {
        if (_check_mask(&it_mask)) {
            memcpy(dst_raw + write_idx * elem_size, it_data.it.ptr, elem_size);
            write_idx++;
        }
        mf_accessor_f32_advance(&it_data, st1);
        mf_accessor_u8_advance(&it_mask, st2);
    }
}

// --- Op: Gather (Random Access) ---
static void op_gather(mf_exec_ctx* ctx, const struct mf_instruction* inst) {
    mf_tensor* dst = &ctx->registers[inst->dest_idx];
    mf_tensor* src_data = &ctx->registers[inst->src1_idx];
    mf_tensor* src_indices = &ctx->registers[inst->src2_idx];
    
    MF_CHECK_DST_VIEW(ctx, dst);
    MF_CHECK_INPUT(ctx, src_data);
    MF_CHECK_INPUT(ctx, src_indices);

    if (!mf_utils_resolve_unary_shape(ctx, dst, src_indices)) return;
    // DType should be set by compiler, but we ensure it matches src_data for safety
    if (dst->info.dtype == MF_DTYPE_UNKNOWN) dst->info.dtype = src_data->info.dtype;

    MF_CHECK_DST_DATA(ctx, dst);

    size_t data_count = mf_tensor_count(src_data);
    size_t out_count = mf_tensor_count(dst);
    size_t elem_size = mf_dtype_size(src_data->info.dtype);
    
    // Iterators for Indices (Can be F32 or I32)
    bool idx_is_f32 = (src_indices->info.dtype == MF_DTYPE_F32);
    mf_accessor_f32 it_idx_f32;
    mf_accessor_i32 it_idx_i32;
    if (idx_is_f32) it_idx_f32 = mf_accessor_f32_begin(src_indices);
    else it_idx_i32 = mf_accessor_i32_begin(src_indices);

    // Raw pointer for destination (Generic copy)
    u8* dst_ptr = (u8*)mf_tensor_data(dst);
    i32 st_dst = MF_GET_STRIDE_D(inst);
    i32 st_idx = MF_GET_STRIDE_S2(inst);

    for (size_t i = 0; i < out_count; ++i) {
        int idx = -1;
        if (idx_is_f32) {
            idx = (int)mf_accessor_f32_get_safe(&it_idx_f32);
            mf_accessor_f32_advance(&it_idx_f32, st_idx);
        } else {
            idx = (int)mf_accessor_i32_get(&it_idx_i32);
            mf_accessor_i32_advance(&it_idx_i32, st_idx);
        }

        u8* target = dst_ptr + (i * st_dst * elem_size);
        if (idx >= 0 && (size_t)idx < data_count) {
            void* src_ptr = mf_tensor_iter_get_at_linear(src_data, (size_t)idx);
            memcpy(target, src_ptr, elem_size);
        } else {
            memset(target, 0, elem_size);
            if (_mf_should_log_error(ctx)) {
                ctx->error = MF_ERROR_OUT_OF_BOUNDS;
                ctx->error_idx = (u32)i;
                MF_LOG_ERROR("Gather OOB: Index %d at batch element %zu. Data size: %zu. Using 0.", 
                             idx, i, data_count);
            }
        }
    }
}

void mf_ops_array_register(mf_op_func* table) {
    table[MF_OP_GATHER] = op_gather;
    table[MF_OP_CUMSUM] = op_cumsum;
    table[MF_OP_COMPRESS] = op_compress;
}