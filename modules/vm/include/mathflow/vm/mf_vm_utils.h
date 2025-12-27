#ifndef MF_VM_UTILS_H
#define MF_VM_UTILS_H

#include <mathflow/vm/mf_vm.h>
#include <mathflow/isa/mf_tensor.h>
#include <math.h>
#include <string.h>

// --- Helper: Shape Resolution (Inline) ---

static inline bool mf_utils_resolve_binary_shape(mf_vm* vm, mf_tensor* dst, const mf_tensor* a, const mf_tensor* b) {
    const mf_tensor* shape_src = (a->size >= b->size) ? a : b;
    // Default: output dtype matches A (usually f32)
    if (dst->dtype == MF_DTYPE_UNKNOWN) dst->dtype = a->dtype;
    return mf_vm_resize_tensor(vm, dst, shape_src->shape, shape_src->ndim);
}

static inline bool mf_utils_resolve_unary_shape(mf_vm* vm, mf_tensor* dst, const mf_tensor* a) {
    if (dst->dtype == MF_DTYPE_UNKNOWN) dst->dtype = a->dtype;
    return mf_vm_resize_tensor(vm, dst, a->shape, a->ndim);
}

// --- Macros: Kernel Definitions ---

// Helper for generic binary ops (C = A op B)
// Supports scalar broadcasting (if size==1)
#define MF_KERNEL_BINARY(NAME, OP) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!mf_utils_resolve_binary_shape(vm, dst, a, b)) return; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); \
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = (a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i]); \
}

// Helper for function-based binary ops (C = func(A, B))
#define MF_KERNEL_BINARY_FUNC(NAME, FUNC) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    if (!mf_utils_resolve_binary_shape(vm, dst, a, b)) return; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); \
    f32* da = (f32*)a->data; f32* db = (f32*)b->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = FUNC((a_s ? da[0] : da[i]), (b_s ? db[0] : db[i])); \
}

// Helper for unary ops (C = func(A))
#define MF_KERNEL_UNARY(NAME, FUNC) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    if (!dst || !a) return; \
    if (!mf_utils_resolve_unary_shape(vm, dst, a)) return; \
    f32* da = (f32*)a->data; f32* dd = (f32*)dst->data; \
    for(size_t i=0; i<dst->size; ++i) dd[i] = FUNC(da[i]); \
}

// Helper for comparison ops (C = A op B), Output is always U8 (bool)
#define MF_KERNEL_COMPARE(NAME, OP) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->dtype = MF_DTYPE_U8; /* Force Bool Output */ \
    if (!mf_utils_resolve_binary_shape(vm, dst, a, b)) return; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); u8* dd = (u8*)dst->data; \
    if (a->dtype == MF_DTYPE_F32) { \
        f32* da = (f32*)a->data; f32* db = (f32*)b->data; \
        for(size_t i=0; i<dst->size; ++i) dd[i] = ((a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i])) ? 1 : 0; \
    } else if (a->dtype == MF_DTYPE_I32) { \
        int32_t* da = (int32_t*)a->data; int32_t* db = (int32_t*)b->data; \
        for(size_t i=0; i<dst->size; ++i) dd[i] = ((a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i])) ? 1 : 0; \
    } \
}

// Helper for logic ops (C = A op B), Input/Output U8
#define MF_KERNEL_LOGIC(NAME, OP) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!dst || !a || !b) return; \
    dst->dtype = MF_DTYPE_U8; \
    if (!mf_utils_resolve_binary_shape(vm, dst, a, b)) return; \
    u8* da = (u8*)a->data; u8* db = (u8*)b->data; u8* dd = (u8*)dst->data; \
    bool a_s = (a->size == 1); bool b_s = (b->size == 1); \
    for(size_t i=0; i<dst->size; ++i) dd[i] = (a_s ? da[0] : da[i]) OP (b_s ? db[0] : db[i]); \
}

#endif // MF_VM_UTILS_H
