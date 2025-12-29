#include <mathflow/vm/mf_vm.h>
#include <mathflow/base/mf_memory.h>
#include <string.h>
#include <stdio.h>

// --- VM API ---

void mf_vm_init(mf_vm* vm, mf_allocator* allocator) {
    memset(vm, 0, sizeof(mf_vm));
    vm->allocator = allocator;
}

void mf_vm_reset(mf_vm* vm, const mf_program* prog, mf_arena* arena) {
    if (!prog) return;
    
    // Allocate Registers (Tensors) in Arena (Metadata only)
    vm->register_count = prog->meta.tensor_count;
    vm->registers = MF_ARENA_PUSH(arena, mf_tensor, vm->register_count);

    // Copy initial state from Program Prototypes
    for (u32 i = 0; i < vm->register_count; ++i) {
        mf_tensor* dst = &vm->registers[i];
        const mf_tensor* src = &prog->tensors[i];
        
        *dst = *src; // Copy metadata
        dst->flags = 0; 
        
        if (src->data) {
            // Constant: Deep copy data to Allocator memory.
            if (vm->allocator) {
                size_t bytes = src->capacity_bytes;
                dst->data = vm->allocator->alloc(vm->allocator, bytes);
                if (dst->data) memcpy(dst->data, src->data, bytes);
                dst->capacity_bytes = bytes;
                dst->flags |= MF_TENSOR_OWNS_DATA | MF_TENSOR_DYNAMIC;
            } else {
                 // Fallback: Point to shared program memory (Read-Only)
                 dst->data = src->data; 
                 // Do not set OWNS_DATA
            }
        } else {
            // Variable: Allocate a small default buffer. Kernels will resize if needed.
            if (vm->allocator) {
                 size_t type_size = mf_dtype_size(src->dtype);
                 if (type_size == 0) type_size = 4; // Default to F32
                 
                 size_t bytes = (src->size > 0) ? src->size * type_size : type_size;
                 
                 dst->data = vm->allocator->alloc(vm->allocator, bytes);
                 if (dst->data) memset(dst->data, 0, bytes);
                 
                 dst->capacity_bytes = bytes;
                 dst->flags |= MF_TENSOR_OWNS_DATA | MF_TENSOR_DYNAMIC;
            } else {
                 dst->data = NULL;
                 dst->capacity_bytes = 0;
                 dst->flags |= MF_TENSOR_DYNAMIC;
            }
        }
    }
}

void mf_vm_shutdown(mf_vm* vm) {
    if (!vm->registers || !vm->allocator) return;
    
    for (u32 i = 0; i < vm->register_count; ++i) {
        mf_tensor* t = &vm->registers[i];
        if (t->data && (t->flags & MF_TENSOR_OWNS_DATA)) {
            vm->allocator->free(vm->allocator, t->data);
            t->data = NULL;
            t->flags &= ~MF_TENSOR_OWNS_DATA;
        }
    }
}

mf_tensor* mf_vm_map_tensor(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (idx >= vm->register_count) return NULL;
    return &vm->registers[idx];
}

bool mf_vm_resize_tensor(mf_vm* vm, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim) {
    if (!vm->allocator) {
        vm->error = MF_ERROR_OOM;
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
            new_ptr = vm->allocator->realloc(vm->allocator, tensor->data, valid_bytes, needed_bytes);
        } else {
            // First alloc or transitioning from static (arena) to dynamic
            new_ptr = vm->allocator->alloc(vm->allocator, needed_bytes);
            if (tensor->data && !(tensor->flags & MF_TENSOR_OWNS_DATA)) {
                // Preserve data if resizing from static
                size_t old_bytes = tensor->size * type_size; 
                size_t copy_bytes = (old_bytes < needed_bytes) ? old_bytes : needed_bytes;
                memcpy(new_ptr, tensor->data, copy_bytes);
            }
        }
        
        if (!new_ptr && needed_bytes > 0) {
            vm->error = MF_ERROR_OOM;
            return false; 
        }
        
        tensor->data = new_ptr;
        tensor->capacity_bytes = needed_bytes;
        tensor->flags |= MF_TENSOR_OWNS_DATA;
    }
    
    return true;
}
