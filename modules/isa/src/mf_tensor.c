#include <mathflow/isa/mf_tensor.h>
#include <string.h>

bool mf_tensor_resize(mf_tensor* tensor, mf_allocator* allocator, const int32_t* new_shape, uint8_t new_ndim) {
    if (!allocator) return false;
    
    // Calculate new size
    size_t new_count = 1;
    for (int i = 0; i < new_ndim; ++i) {
        new_count *= new_shape[i];
    }
    if (new_ndim == 0) new_count = 1; // Scalar
    
    size_t type_size = mf_dtype_size(tensor->dtype);
    size_t needed_bytes = new_count * type_size;
    
    // Update shape meta
    tensor->ndim = new_ndim;
    memcpy(tensor->shape, new_shape, sizeof(int32_t) * MF_MAX_DIMS); 
    tensor->size = new_count;
    
    // Check capacity
    if (needed_bytes > tensor->capacity_bytes) {
        void* new_ptr = NULL;
        if (tensor->data && (tensor->flags & MF_TENSOR_OWNS_DATA)) {
            size_t valid_bytes = tensor->size * type_size; // Previous valid size
            // Note: realloc might copy garbage if we increased size, but we care about preserving valid data.
            // Actually, we usually want to preserve the OLD data up to min(old_size, new_size).
            // But here we rely on allocator->realloc semantics.
            new_ptr = allocator->realloc(allocator, tensor->data, valid_bytes, needed_bytes);
        } else {
            // First alloc or transitioning from static to dynamic
            new_ptr = allocator->alloc(allocator, needed_bytes);
            if (tensor->data && !(tensor->flags & MF_TENSOR_OWNS_DATA)) {
                // Preserve data if resizing from static
                size_t old_bytes = tensor->size * type_size; 
                size_t copy_bytes = (old_bytes < needed_bytes) ? old_bytes : needed_bytes;
                memcpy(new_ptr, tensor->data, copy_bytes);
            }
        }
        
        if (!new_ptr && needed_bytes > 0) {
            return false; 
        }
        
        tensor->data = new_ptr;
        tensor->capacity_bytes = needed_bytes;
        tensor->flags |= MF_TENSOR_OWNS_DATA;
    }
    
    return true;
}

bool mf_tensor_clone(mf_tensor* dst, const mf_tensor* src, mf_allocator* allocator) {
    if (!dst || !src) return false;

    // Base Metadata Copy
    *dst = *src;
    dst->flags = 0; // Reset flags (don't inherit ownership yet)
    
    // If no allocator provided, create a shallow alias (unsafe for mutable usage but valid for read-only)
    if (!allocator) {
        dst->data = src->data;
        dst->capacity_bytes = 0; // Not owned
        dst->flags |= MF_TENSOR_DYNAMIC; // Treat as dynamic? Or maybe just external.
        return true;
    }

    // Determine allocation size
    size_t type_size = mf_dtype_size(src->dtype);
    if (type_size == 0) type_size = 4; // Fallback
    
    size_t bytes = src->capacity_bytes;
    if (bytes == 0) {
        bytes = (src->size > 0) ? src->size * type_size : type_size;
    }
    
    // Allocate
    dst->data = allocator->alloc(allocator, bytes);
    if (!dst->data && bytes > 0) {
        return false; // OOM
    }

    dst->capacity_bytes = bytes;
    dst->flags |= MF_TENSOR_OWNS_DATA | MF_TENSOR_DYNAMIC;

    // Initialize Content
    if (src->data) {
        memcpy(dst->data, src->data, bytes);
    } else {
        memset(dst->data, 0, bytes);
    }

    return true;
}
