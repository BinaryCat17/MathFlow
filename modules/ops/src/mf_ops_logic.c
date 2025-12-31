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

static void op_not(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    if (!dst || !a) return; 
    
    dst->info.dtype = MF_DTYPE_U8;
    if (!mf_utils_resolve_unary_shape(ctx, dst, a)) return;
    
    u8* da = (u8*)mf_tensor_data(a); 
    u8* dd = (u8*)mf_tensor_data(dst);
    size_t count = mf_tensor_count(dst);
    for(size_t i=0; i<count; ++i) dd[i] = !da[i];
}

// --- Selection ---

static void op_select(mf_exec_ctx* ctx, const mf_instruction* inst) {
    mf_tensor* dst = mf_exec_ctx_map_tensor(ctx, inst->dest_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_exec_ctx_map_tensor(ctx, inst->src1_idx, MF_ACCESS_READ);
    mf_tensor* true_val = mf_exec_ctx_map_tensor(ctx, inst->src2_idx, MF_ACCESS_READ);
    mf_tensor* false_val = mf_exec_ctx_map_tensor(ctx, inst->src3_idx, MF_ACCESS_READ);
    
    if (!dst || !cond || !true_val || !false_val) return;
    
    // Enforce type from True Branch
    dst->info.dtype = true_val->info.dtype; 
    
    if (!mf_utils_resolve_ternary_shape(ctx, dst, cond, true_val, false_val)) return;
    
    u8* c = (u8*)mf_tensor_data(cond);
    f32* c_f32 = (f32*)c;
    // Note: We use raw bytes here because Select doesn't care about type, it just copies bytes.
    // However, strides matter! 
    // Optimization: If simple types and contiguous, we can use simple loop.
    
    // For now, let's assume contiguous/resolved via resolve_ternary_shape (which does broadcasting).
    // We need element pointers.
    
    size_t es = mf_dtype_size(dst->info.dtype);
    u8* d = (u8*)mf_tensor_data(dst);
    u8* vt = (u8*)mf_tensor_data(true_val);
    u8* vf = (u8*)mf_tensor_data(false_val);
    
    size_t sz_cond = mf_tensor_count(cond);
    size_t sz_vt = mf_tensor_count(true_val);
    size_t sz_vf = mf_tensor_count(false_val);
    size_t sz_dst = mf_tensor_count(dst);
    
    bool c_s = (sz_cond == 1);
    bool vt_s = (sz_vt == 1);
    bool vf_s = (sz_vf == 1);
    
    for(size_t i=0; i<sz_dst; ++i) {
        bool condition = false;
        if (cond->info.dtype == MF_DTYPE_F32) {
            condition = (c_f32[c_s ? 0 : i] != 0.0f);
        } else {
            condition = (c[c_s ? 0 : i] != 0);
        }

        if (condition) {
            memcpy(d + i*es, vt + (vt_s ? 0 : i*es), es);
        } else {
            memcpy(d + i*es, vf + (vf_s ? 0 : i*es), es);
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
    
    table[MF_OP_SELECT] = op_select;
}
