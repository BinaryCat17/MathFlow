#ifndef MF_TENSOR_ITER_H
#define MF_TENSOR_ITER_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/base/mf_log.h>

/**
 * mf_tensor_iter is a lightweight N-dimensional iterator.
 * It tracks current multi-dimensional index and calculates physical pointer.
 */
typedef struct {
    void* ptr;           // Current pointer to data
    void* start;         // Valid range start (usually tensor_data)
    void* limit;         // Valid range end (start + size_bytes)
    size_t element_size; // Bytes per element
    
    int32_t indices[MF_MAX_DIMS];
    const mf_tensor* tensor;
    
    bool is_contiguous;
} mf_tensor_iter;

static inline mf_tensor_iter mf_tensor_iter_begin(const mf_tensor* t) {
    mf_tensor_iter it;
    it.tensor = t;
    it.element_size = mf_dtype_size(t->info.dtype);
    it.ptr = mf_tensor_data(t);
    it.start = it.ptr;
    
    // Limit is relative to the buffer to ensure absolute safety
    if (t->buffer) {
        it.limit = (uint8_t*)t->buffer->data + t->buffer->size_bytes;
    } else {
        it.limit = it.ptr; // Invalid tensor
    }

    it.is_contiguous = mf_tensor_is_contiguous(t);
    for (int i = 0; i < MF_MAX_DIMS; ++i) it.indices[i] = 0;
    return it;
}

static inline void mf_tensor_iter_next(mf_tensor_iter* it) {
    if (it->is_contiguous) {
        it->ptr = (uint8_t*)it->ptr + it->element_size;
    } else {
        // N-Dimensional step
        const mf_tensor* t = it->tensor;
        for (int i = t->info.ndim - 1; i >= 0; --i) {
            it->indices[i]++;
            if (it->indices[i] < t->info.shape[i]) {
                // Stay in this dimension, move by stride
                it->ptr = (uint8_t*)it->ptr + (t->info.strides[i] * it->element_size);
                goto check_bounds;
            }
            
            // Overflow: reset this dimension and carry to next
            it->ptr = (uint8_t*)it->ptr - ((t->info.shape[i] - 1) * t->info.strides[i] * it->element_size);
            it->indices[i] = 0;
        }
    }

check_bounds:
    if (it->ptr > it->limit || (it->ptr < it->start && it->tensor->info.ndim > 0)) {
        MF_LOG_FATAL("Tensor iterator out of bounds! Ptr: %p, Range: [%p, %p]", 
                     it->ptr, it->start, it->limit);
    }
}

static inline void mf_tensor_iter_advance(mf_tensor_iter* it, i32 step) {
    if (step == 1) {
        mf_tensor_iter_next(it);
        return;
    }
    if (step == 0) return;
    
    if (it->is_contiguous) {
        it->ptr = (uint8_t*)it->ptr + (size_t)step * it->element_size;
    } else {
        for (i32 i = 0; i < step; ++i) mf_tensor_iter_next(it);
        return;
    }

    if (it->ptr > it->limit || (it->ptr < it->start && it->tensor->info.ndim > 0)) {
        MF_LOG_FATAL("Tensor iterator out of bounds after advance! Ptr: %p, Range: [%p, %p], Step: %d", 
                     it->ptr, it->start, it->limit, step);
    }
}

// Helper for broadcasting/looping with potential scalar/mismatch shapes
static inline void* mf_tensor_iter_get_at_linear(const mf_tensor* t, size_t linear_index) {
    void* result_ptr = NULL;
    size_t el_size = mf_dtype_size(t->info.dtype);

    if (mf_tensor_is_contiguous(t)) {
        result_ptr = (uint8_t*)mf_tensor_data(t) + linear_index * el_size;
    } else {
        // Fallback: calculate multi-dim index from linear
        int32_t indices[MF_MAX_DIMS];
        size_t count = linear_index;
        for (int i = t->info.ndim - 1; i >= 0; --i) {
            int32_t dim = t->info.shape[i];
            if (dim <= 0) {
                indices[i] = 0;
            } else {
                indices[i] = (int32_t)(count % dim);
                count /= dim;
            }
        }
        result_ptr = (uint8_t*)mf_tensor_data(t) + mf_tensor_get_offset(t, indices) * el_size;
    }

    // Strict validation for random access
    void* start = mf_tensor_data(t);
    void* limit = (uint8_t*)t->buffer->data + t->buffer->size_bytes;
    
    if (result_ptr < start || (uint8_t*)result_ptr + el_size > (uint8_t*)limit) {
        MF_LOG_FATAL("Tensor random access out of bounds! Index: %zu, Ptr: %p, Range: [%p, %p]", 
                     linear_index, result_ptr, start, limit);
    }
    
    return result_ptr;
}

#endif // MF_TENSOR_ITER_H
