#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_tensor.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- Memory Helper ---
// Basic allocator for outputs. Leak warning: no free() yet.
static void ensure_shape(mf_tensor* dst, const mf_tensor* src_shape) {
    bool same_shape = (dst->ndim == src_shape->ndim);
    if (same_shape) {
        for(int i=0; i<dst->ndim; ++i) if(dst->shape[i] != src_shape->shape[i]) same_shape = false;
    }

    if (!dst->data || !same_shape) {
        // Reallocate
        if (dst->data) free(dst->data); // Assume malloc
        
        dst->dtype = src_shape->dtype;
        dst->ndim = src_shape->ndim;
        memcpy(dst->shape, src_shape->shape, sizeof(dst->shape));
        dst->size = src_shape->size;
        
        dst->data = malloc(dst->size * mf_dtype_size(dst->dtype));
    }
}

// --- Kernel: Binary Math ---

#define OP_KERNEL_BINARY(NAME, OP) \
static void op_##NAME(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) { \
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE); \
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ); \
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ); \
    if (!a || !b) return; \
    \
    bool a_scalar = (a->size == 1); \
    bool b_scalar = (b->size == 1); \
    \
    if (a_scalar && b_scalar) { \
        ensure_shape(dst, a); \
        f32 val_a = ((f32*)a->data)[0]; \
        f32 val_b = ((f32*)b->data)[0]; \
        ((f32*)dst->data)[0] = val_a OP val_b; \
    } \
    else if (!a_scalar && !b_scalar) { \
        /* Element-wise (assume same shape) */ \
        ensure_shape(dst, a); \
        f32* da = (f32*)a->data; \
        f32* db = (f32*)b->data; \
        f32* dd = (f32*)dst->data; \
        for(size_t i=0; i<a->size; ++i) dd[i] = da[i] OP db[i]; \
    } \
    else if (a_scalar) { \
        /* Broadcast A -> B */ \
        ensure_shape(dst, b); \
        f32 val_a = ((f32*)a->data)[0]; \
        f32* db = (f32*)b->data; \
        f32* dd = (f32*)dst->data; \
        for(size_t i=0; i<b->size; ++i) dd[i] = val_a OP db[i]; \
    } \
    else { \
        /* Broadcast B -> A */ \
        ensure_shape(dst, a); \
        f32* da = (f32*)a->data; \
        f32 val_b = ((f32*)b->data)[0]; \
        f32* dd = (f32*)dst->data; \
        for(size_t i=0; i<a->size; ++i) dd[i] = da[i] OP val_b; \
    } \
}

OP_KERNEL_BINARY(add, +)
OP_KERNEL_BINARY(sub, -)
OP_KERNEL_BINARY(mul, *)
OP_KERNEL_BINARY(div, /)

// --- Kernel: Binary Func (Min/Max) ---
// Need function pointers instead of operators
static void op_min(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!a || !b) return;
    ensure_shape(dst, (a->size > b->size) ? a : b);
    
    f32* da = (f32*)a->data;
    f32* db = (f32*)b->data;
    f32* dd = (f32*)dst->data;
    size_t n = dst->size;
    
    bool a_scal = (a->size == 1);
    bool b_scal = (b->size == 1);
    
    for(size_t i=0; i<n; ++i) {
        f32 va = a_scal ? da[0] : da[i];
        f32 vb = b_scal ? db[0] : db[i];
        dd[i] = (va < vb) ? va : vb;
    }
}

static void op_max(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* a = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* b = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    if (!a || !b) return;
    ensure_shape(dst, (a->size > b->size) ? a : b);
    
    f32* da = (f32*)a->data;
    f32* db = (f32*)b->data;
    f32* dd = (f32*)dst->data;
    size_t n = dst->size;
    bool a_scal = (a->size == 1);
    bool b_scal = (b->size == 1);
    
    for(size_t i=0; i<n; ++i) {
        f32 va = a_scal ? da[0] : da[i];
        f32 vb = b_scal ? db[0] : db[i];
        dd[i] = (va > vb) ? va : vb;
    }
}

// --- Kernel: Where (Select) ---
static void op_where_true(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* val = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    
    // Logic: Dest = Val where Cond is True. Else keep Dest.
    // Ensure Dest is initialized to Shape of Val
    ensure_shape(dst, val);
    
    u8* c = (u8*)cond->data; // Assume Bool U8
    f32* v = (f32*)val->data;
    f32* d = (f32*)dst->data;
    
    bool c_scal = (cond->size == 1);
    
    for(size_t i=0; i<dst->size; ++i) {
        if (c[c_scal ? 0 : i]) d[i] = v[i];
    }
}

static void op_where_false(mf_vm* vm, u16 dst_idx, u16 src1_idx, u16 src2_idx) {
    mf_tensor* dst = mf_vm_map_tensor(vm, dst_idx, MF_ACCESS_WRITE);
    mf_tensor* cond = mf_vm_map_tensor(vm, src1_idx, MF_ACCESS_READ);
    mf_tensor* val = mf_vm_map_tensor(vm, src2_idx, MF_ACCESS_READ);
    
    ensure_shape(dst, val);
    
    u8* c = (u8*)cond->data;
    f32* v = (f32*)val->data;
    f32* d = (f32*)dst->data;
    
    bool c_scal = (cond->size == 1);
    
    for(size_t i=0; i<dst->size; ++i) {
        if (!c[c_scal ? 0 : i]) d[i] = v[i];
    }
}

// --- Init ---

void mf_backend_cpu_init(mf_backend_dispatch_table* table) {
    memset(table, 0, sizeof(mf_backend_dispatch_table));
    
    table->op_table[MF_OP_ADD] = op_add;
    table->op_table[MF_OP_SUB] = op_sub;
    table->op_table[MF_OP_MUL] = op_mul;
    table->op_table[MF_OP_DIV] = op_div;
    
    table->op_table[MF_OP_MIN] = op_min;
    table->op_table[MF_OP_MAX] = op_max;
    
    table->op_table[MF_OP_WHERE_TRUE] = op_where_true;
    table->op_table[MF_OP_WHERE_FALSE] = op_where_false;
}