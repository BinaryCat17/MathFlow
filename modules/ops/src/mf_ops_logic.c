#include <mathflow/ops/mf_ops_core.h>
#include "mf_kernel_utils.h"
#include <mathflow/isa/mf_opcodes.h>
#include "mf_ops_internal.h"
#include <string.h>

// --- Comparison ---

MF_KERNEL_COMPARE(less, <)
MF_KERNEL_COMPARE(greater, >)
MF_KERNEL_COMPARE(equal, ==)
MF_KERNEL_COMPARE(nequal, !=)
MF_KERNEL_COMPARE(lequal, <=)
MF_KERNEL_COMPARE(gequal, >=)

// --- Logic ---

MF_KERNEL_LOGIC(and, &&)
MF_KERNEL_LOGIC(or, ||)

static void op_not(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    (void)src2_idx;
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return; 
    
    dst->info.dtype = MF_DTYPE_U8;
    if (!mf_utils_resolve_unary_shape(ctx, dst, a)) return;
    
    u8* da = (u8*)mf_tensor_data(a); 
    u8* dd = (u8*)mf_tensor_data(dst);
    size_t count = mf_tensor_count(dst);
    for(size_t i=0; i<count; ++i) dd[i] = !da[i];
}

// --- Selection ---

static void op_where_true(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* val = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    if (!dst || !cond || !val) return;
    
    dst->info.dtype = val->info.dtype; // Enforce type
    if (!mf_utils_resolve_binary_shape(ctx, dst, cond, val)) return;
    
    u8* c = (u8*)mf_tensor_data(cond); 
    u8* v = (u8*)mf_tensor_data(val); 
    u8* d = (u8*)mf_tensor_data(dst); 
    
    size_t es = mf_dtype_size(val->info.dtype);
    size_t sz_cond = mf_tensor_count(cond);
    size_t sz_val = mf_tensor_count(val);
    size_t sz_dst = mf_tensor_count(dst);
    
    bool c_s = (sz_cond == 1); 
    bool v_s = (sz_val == 1);
    
    for(size_t i=0; i<sz_dst; ++i) {
        if (c[c_s ? 0 : i]) {
            memcpy(d + i*es, v + (v_s ? 0 : i*es), es);
        }
    }
}

static void op_where_false(mf_exec_ctx* ctx, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_exec_ctx_map_tensor(ctx, src1_idx, MF_ACCESS_READ);
    mf_tensor* val = mf_exec_ctx_map_tensor(ctx, src2_idx, MF_ACCESS_READ);
    if (!dst || !cond || !val) return;

    dst->info.dtype = val->info.dtype;
    if (!mf_utils_resolve_binary_shape(ctx, dst, cond, val)) return;

    u8* c = (u8*)mf_tensor_data(cond); 
    u8* v = (u8*)mf_tensor_data(val); 
    u8* d = (u8*)mf_tensor_data(dst); 
    
    size_t es = mf_dtype_size(val->info.dtype);
    size_t sz_cond = mf_tensor_count(cond);
    size_t sz_val = mf_tensor_count(val);
    size_t sz_dst = mf_tensor_count(dst);
    
    bool c_s = (sz_cond == 1); 
    bool v_s = (sz_val == 1);
    
    for(size_t i=0; i<sz_dst; ++i) {
        if (!c[c_s ? 0 : i]) {
            memcpy(d + i*es, v + (v_s ? 0 : i*es), es);
        }
    }
}

void mf_ops_register_logic(mf_op_func* table) {
    table[MF_OP_LESS] = op_less; 
    table[MF_OP_GREATER] = op_greater; 
    table[MF_OP_EQUAL] = op_equal;
    table[MF_OP_NEQUAL] = op_nequal; 
    table[MF_OP_LEQUAL] = op_lequal; 
    table[MF_OP_GEQUAL] = op_gequal;
    
    table[MF_OP_AND] = op_and; 
    table[MF_OP_OR] = op_or; 
    table[MF_OP_NOT] = op_not;
    
    table[MF_OP_WHERE_TRUE] = op_where_true; 
    table[MF_OP_WHERE_FALSE] = op_where_false;
}