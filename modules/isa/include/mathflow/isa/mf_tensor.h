#ifndef MF_TENSOR_H
#define MF_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <mathflow/base/mf_types.h>

// Max dimensions supported (Rank)
#define MF_MAX_DIMS 8

// --- Data Types ---

typedef enum {
    MF_DTYPE_UNKNOWN = 0,
    
    MF_DTYPE_F32,   // Standard float
    MF_DTYPE_I32,   // Integer / String ID
    MF_DTYPE_U8,    // Byte / Bool
    
    MF_DTYPE_COUNT
} mf_dtype;

// --- Tensor Structure ---

typedef struct {
    mf_dtype dtype;
    
    // Shape: [batch, height, width, channels] etc.
    // e.g., Scalar = {0}, Vec3 = {3}, Mat4 = {4,4}
    uint8_t ndim; // Number of dimensions (Rank)
    int32_t shape[MF_MAX_DIMS]; 
    
    // Strides: Steps in memory to reach next element in each dim.
    // Allows slicing/transposing without data copy.
    int32_t strides[MF_MAX_DIMS]; 

    // Raw Data Pointer
    // This memory is owned by the VM/Arena, not the struct itself.
    void* data;
    
    // Total elements count (cached product of shape)
    size_t size; 

    // Allocated size in bytes (for dynamic resizing)
    size_t capacity_bytes;

    // Flags
    u32 flags;
} mf_tensor;

#define MF_TENSOR_OWNS_DATA 1
#define MF_TENSOR_DYNAMIC   2  // Can be resized

// --- Helper Functions (Inline) ---

static inline size_t mf_dtype_size(mf_dtype type) {
    switch (type) {
        case MF_DTYPE_F32: return 4;
        case MF_DTYPE_I32: return 4;
        case MF_DTYPE_U8:  return 1;
        default: return 0;
    }
}

// Check if tensor is a scalar (Rank 0)
static inline bool mf_tensor_is_scalar(const mf_tensor* t) {
    return t->ndim == 0;
}

// Check if shapes are identical
static inline bool mf_tensor_same_shape(const mf_tensor* a, const mf_tensor* b) {
    if (a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

static inline size_t mf_tensor_size_bytes(const mf_tensor* t) {
    return t->size * mf_dtype_size(t->dtype);
}

#endif // MF_TENSOR_H
