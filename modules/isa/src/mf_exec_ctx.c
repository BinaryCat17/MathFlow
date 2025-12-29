#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_memory.h>
#include <string.h>

void mf_exec_ctx_init(mf_exec_ctx* ctx, mf_tensor* registers, size_t reg_count, mf_allocator* allocator) {
    memset(ctx, 0, sizeof(mf_exec_ctx));
    ctx->registers = registers;
    ctx->register_count = reg_count;
    ctx->allocator = allocator;
}

mf_tensor* mf_exec_ctx_map_tensor(mf_exec_ctx* ctx, u16 idx, mf_access_mode mode) {
    (void)mode;
    if (idx >= ctx->register_count) return NULL;
    return &ctx->registers[idx];
}

bool mf_exec_ctx_resize_tensor(mf_exec_ctx* ctx, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim) {
    if (!ctx->allocator) {
        ctx->error = MF_ERROR_OOM;
        return false;
    }
    
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
            size_t valid_bytes = tensor->size * type_size;
            new_ptr = ctx->allocator->realloc(ctx->allocator, tensor->data, valid_bytes, needed_bytes);
        } else {
            new_ptr = ctx->allocator->alloc(ctx->allocator, needed_bytes);
            if (tensor->data && !(tensor->flags & MF_TENSOR_OWNS_DATA)) {
                size_t old_bytes = tensor->size * type_size; 
                size_t copy_bytes = (old_bytes < needed_bytes) ? old_bytes : needed_bytes;
                memcpy(new_ptr, tensor->data, copy_bytes);
            }
        }
        
        if (!new_ptr && needed_bytes > 0) {
            ctx->error = MF_ERROR_OOM;
            return false; 
        }
        
        tensor->data = new_ptr;
        tensor->capacity_bytes = needed_bytes;
        tensor->flags |= MF_TENSOR_OWNS_DATA;
    }
    
    return true;
}