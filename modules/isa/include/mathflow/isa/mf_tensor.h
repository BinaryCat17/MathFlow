#ifndef MF_TENSOR_H
#define MF_TENSOR_H

#include <mathflow/base/mf_types.h>
#include <mathflow/base/mf_buffer.h>
#include <mathflow/base/mf_memory.h>

// --- Tensor Structure ---

// A Tensor is a VIEW into a buffer.
typedef struct {
    mf_type_info info;   // Metadata: Shape, Strides, Type
    
    mf_buffer* buffer;   // Storage: Pointer to data owner
    size_t byte_offset;  // Offset in bytes from buffer->data
} mf_tensor;

// --- Helper Functions (Inline) ---

// Get raw pointer to data start (with offset applied)
static inline void* mf_tensor_data(const mf_tensor* t) {
    if (!t || !t->buffer || !t->buffer->data) return NULL;
    return (uint8_t*)t->buffer->data + t->byte_offset;
}

static inline bool mf_tensor_is_valid(const mf_tensor* t) {
    return t && t->buffer && t->buffer->data;
}

static inline bool mf_tensor_is_scalar(const mf_tensor* t) {
    return t->info.ndim == 0;
}

static inline size_t mf_tensor_count(const mf_tensor* t) {
    if (!t || t->info.ndim == 0) return 1;
    size_t count = 1;
    for(int i=0; i < t->info.ndim; ++i) {
        int32_t dim = t->info.shape[i];
        count *= (dim > 0 ? (size_t)dim : 0);
    }
    return count;
}

static inline size_t mf_tensor_size_bytes(const mf_tensor* t) {
    return mf_tensor_count(t) * mf_dtype_size(t->info.dtype);
}

static inline bool mf_tensor_same_shape(const mf_tensor* a, const mf_tensor* b) {
    if (a->info.ndim != b->info.ndim) return false;
    for (int i = 0; i < a->info.ndim; ++i) {
        if (a->info.shape[i] != b->info.shape[i]) return false;
    }
    return true;
}

// Check if the tensor data is contiguous in memory
static inline bool mf_tensor_is_contiguous(const mf_tensor* t) {
    if (t->info.ndim <= 1) return true;
    int32_t stride = 1;
    for (int i = t->info.ndim - 1; i >= 0; --i) {
        if (t->info.strides[i] != stride) return false;
        stride *= t->info.shape[i];
    }
    return true;
}

// Calculate linear element offset from indices [i, j, k, ...]
static inline size_t mf_tensor_get_offset(const mf_tensor* t, const int32_t* indices) {
    size_t offset = 0;
    for (int i = 0; i < t->info.ndim; ++i) {
        offset += indices[i] * t->info.strides[i];
    }
    return offset;
}

// --- Tensor Operations ---

// Init tensor view pointing to an existing buffer
void mf_tensor_init(mf_tensor* tensor, mf_buffer* buf, const mf_type_info* info, size_t offset);

// Allocates a NEW buffer and sets up the tensor view to point to it (Offset 0)
bool mf_tensor_alloc(mf_tensor* tensor, mf_allocator* alloc, const mf_type_info* info);

// Resizes the underlying buffer (reallocation) OR creates a new buffer
// NOTE: This modifies the 'buffer' field. If the buffer was shared, this might detach logic.
bool mf_tensor_resize(mf_tensor* tensor, mf_allocator* allocator, const mf_type_info* new_info);

// Deep copy: Src -> Dst (allocates Dst if needed)
bool mf_tensor_copy_data(mf_tensor* dst, const mf_tensor* src);

// Shallow copy: Dst becomes a view of Src
void mf_tensor_view(mf_tensor* dst, const mf_tensor* src);

// Zero-Copy View Operations (O(1))
// Create a view into a subset of elements (1D slice for now, modifies byte_offset and shape)
bool mf_tensor_slice(mf_tensor* dst, const mf_tensor* src, size_t start_element, size_t count);

// Create a view with a different shape (must have same total element count)
bool mf_tensor_reshape(mf_tensor* dst, const mf_tensor* src, const int32_t* new_shape, int ndim);

// Create a view with swapped dimensions (modifies strides)
bool mf_tensor_transpose(mf_tensor* dst, const mf_tensor* src);

// --- Debugging ---

/**
 * @brief Prints tensor metadata and contents to stdout.
 * @param name Optional label for the tensor.
 * @param t The tensor to print.
 */
void mf_tensor_print(const char* name, const mf_tensor* t);

#endif // MF_TENSOR_H