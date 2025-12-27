#include <mathflow/ops_array/mf_ops_array.h>
#include <mathflow/vm/mf_vm_utils.h>
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>

// --- Op: Range (Iota) ---
// Src1: Scalar count (e.g. 5)
// Dest: Vector [0, 1, 2, 3, 4]
static void op_range(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* count_tensor = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
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
    if (!mf_vm_resize_tensor(vm, dst, shape, 1)) return;

    // Fill
    f32* d = (f32*)dst->data;
    for (int i = 0; i < count; ++i) {
        d[i] = (f32)i;
    }
}

// --- Op: CumSum (Prefix Sum) ---
// Src1: Vector [10, 20, 30]
// Dest: Vector [10, 30, 60]
static void op_cumsum(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* src = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    if (!dst || !src) return;

    // Shape matches source
    if (!mf_utils_resolve_unary_shape(vm, dst, src)) return;

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

// --- Registration ---
void mf_ops_array_register(mf_backend_dispatch_table* table) {
    table->op_table[MF_OP_RANGE] = op_range;
    table->op_table[MF_OP_CUMSUM] = op_cumsum;
}
