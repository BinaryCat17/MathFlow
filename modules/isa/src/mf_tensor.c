#include <mathflow/isa/mf_tensor.h>
#include <mathflow/base/mf_log.h>
#include <string.h>

void mf_tensor_init(mf_tensor* tensor, mf_buffer* buf, const mf_type_info* info, size_t offset) {
    if (!tensor) return;
    if (info) tensor->info = *info;
    else memset(&tensor->info, 0, sizeof(mf_type_info));
    
    tensor->buffer = buf;
    tensor->byte_offset = offset;
}

bool mf_tensor_alloc(mf_tensor* tensor, mf_allocator* alloc, const mf_type_info* info) {
    if (!tensor || !alloc || !info) return false;
    
    tensor->info = *info;
    tensor->byte_offset = 0;
    
    // Allocate the mf_buffer structure itself
    mf_buffer* buf = (mf_buffer*)alloc->alloc(alloc, sizeof(mf_buffer));
    if (!buf) return false;
    
    size_t size_bytes = mf_tensor_size_bytes(tensor);
    if (!mf_buffer_alloc(buf, alloc, size_bytes)) {
        alloc->free(alloc, buf);
        return false;
    }
    
    tensor->buffer = buf;
    return true;
}

bool mf_tensor_resize(mf_tensor* tensor, mf_allocator* allocator, const mf_type_info* new_info) {
    if (!tensor || !allocator || !new_info) return false;
    
    size_t new_size_bytes = 1;
    for(int i=0; i<new_info->ndim; ++i) new_size_bytes *= (new_info->shape[i] > 0 ? new_info->shape[i] : 1);
    new_size_bytes *= mf_dtype_size(new_info->dtype);
    
    // Update metadata
    tensor->info = *new_info;
    
    // Check if we need to realloc
    if (!tensor->buffer) {
        return mf_tensor_alloc(tensor, allocator, new_info);
    }
    
    if (tensor->buffer->size_bytes < new_size_bytes) {
        void* new_data = allocator->alloc(allocator, new_size_bytes);
        if (!new_data) return false;
        
        memset(new_data, 0, new_size_bytes);
        
        if (tensor->buffer->data) {
            size_t copy_size = tensor->buffer->size_bytes;
            if (copy_size > new_size_bytes) copy_size = new_size_bytes;
            memcpy(new_data, tensor->buffer->data, copy_size);
            
            if ((tensor->buffer->flags & MF_BUFFER_OWNS_DATA) && tensor->buffer->alloc) {
                tensor->buffer->alloc->free(tensor->buffer->alloc, tensor->buffer->data);
            }
        }
        
        tensor->buffer->data = new_data;
        tensor->buffer->size_bytes = new_size_bytes;
        tensor->buffer->alloc = allocator;
        tensor->buffer->flags |= MF_BUFFER_OWNS_DATA;
    }
    
    return true;
}

bool mf_tensor_copy_data(mf_tensor* dst, const mf_tensor* src) {
    if (!dst || !src) return false;
    
    void* dst_ptr = mf_tensor_data(dst);
    void* src_ptr = mf_tensor_data(src);
    
    if (!dst_ptr || !src_ptr) return false;
    
    size_t bytes = mf_tensor_size_bytes(dst);
    size_t src_bytes = mf_tensor_size_bytes(src);
    
    // For now strict size match
    if (bytes != src_bytes) return false; 
    
    memcpy(dst_ptr, src_ptr, bytes);
    return true;
}

void mf_tensor_view(mf_tensor* dst, const mf_tensor* src) {
    if (!dst || !src) return;
    *dst = *src; // Copy struct (info + buffer ptr + offset)
}

bool mf_tensor_slice(mf_tensor* dst, const mf_tensor* src, size_t start_element, size_t count) {
    if (!dst || !src) return false;
    
    // Validations
    if (!mf_tensor_is_valid(src)) {
        MF_LOG_ERROR("Tensor Slice: Source tensor is invalid.");
        return false;
    }
    size_t src_count = mf_tensor_count(src);
    if (start_element + count > src_count) {
        MF_LOG_ERROR("Tensor Slice: Out of bounds. Start %zu + Count %zu > Source Count %zu", 
            start_element, count, src_count);
        return false;
    }

    // Create Base View
    mf_tensor_view(dst, src);
    
    // Modify
    size_t elem_size = mf_dtype_size(src->info.dtype);
    dst->byte_offset += start_element * elem_size;
    
    // Flatten shape to 1D for now (or keep dim 0?)
    // "View of Input[Start:End]" usually implies a flat slice or slicing dim 0.
    // If we assume flat slice:
    dst->info.ndim = 1;
    dst->info.shape[0] = (int32_t)count;
    dst->info.strides[0] = 1; // Contiguous
    
    return true;
}

bool mf_tensor_reshape(mf_tensor* dst, const mf_tensor* src, const int32_t* new_shape, int ndim) {
    if (!dst || !src || ndim > MF_MAX_DIMS) return false;
    
    // Count check
    size_t current_count = mf_tensor_count(src);
    size_t new_count = 1;
    for(int i=0; i<ndim; ++i) new_count *= new_shape[i];
    
    if (current_count != new_count) {
        MF_LOG_ERROR("Tensor Reshape: Count mismatch. Current %zu vs New %zu", current_count, new_count);
        return false;
    }
    
    // Create Base View
    mf_tensor_view(dst, src);
    
    // Modify Metadata
    mf_type_info_init_contiguous(&dst->info, src->info.dtype, new_shape, (uint8_t)ndim);
    
    return true;
}

bool mf_tensor_transpose(mf_tensor* dst, const mf_tensor* src) {
    if (!dst || !src) return false;
    
    // Only support 2D transpose for now
    if (src->info.ndim != 2) {
        MF_LOG_ERROR("Tensor Transpose: Only 2D tensors supported, got %dD", src->info.ndim);
        return false;
    }
    
    mf_tensor_view(dst, src);
    
    // Swap Shape
    dst->info.shape[0] = src->info.shape[1];
    dst->info.shape[1] = src->info.shape[0];
    
    // Swap Strides?
    // NOTE: MathFlow assumes row-major contiguity in many ops. 
    // Just swapping metadata strides works for valid "Views", but
    // if an Op assumes contiguous memory (like memcpy), it will fail.
    // For Phase 22, we are introducing this. Ops must be updated to respect strides.
    // Current Ops (mf_ops_matrix.c) do NOT respect strides yet.
    // So this is dangerous without Op updates.
    // BUT: The goal of Phase 22 is to Enable it.
    
    // Let's implement it correctly for the Tensor struct.
    dst->info.strides[0] = src->info.strides[1];
    dst->info.strides[1] = src->info.strides[0];
    
    return true;
}
