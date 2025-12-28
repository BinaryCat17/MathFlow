#include <mathflow/vm/mf_vm.h>
#include <mathflow/base/mf_memory.h>
#include <string.h>

// --- Context API ---

void mf_context_init(mf_context* ctx, const mf_program* prog, mf_backend_dispatch_table* backend) {
    memset(ctx, 0, sizeof(mf_context));
    ctx->code = prog->code;
    ctx->code_count = prog->meta.instruction_count;
    ctx->symbols = prog->symbols;
    ctx->symbol_count = prog->meta.symbol_count;
    ctx->tensor_prototypes = prog->tensors;
    ctx->register_count = prog->meta.tensor_count;
    ctx->backend = backend;
}

// --- VM API ---

void mf_vm_init(mf_vm* vm, const mf_context* ctx, mf_allocator* allocator) {
    memset(vm, 0, sizeof(mf_vm));
    vm->ctx = ctx;
    vm->allocator = allocator;
}

void mf_vm_reset(mf_vm* vm, mf_arena* arena) {
    if (!vm->ctx) return;
    
    // Allocate Registers (Tensors) in Arena (Metadata only)
    // If registers were already allocated, we might want to free their data first?
    // mf_vm_shutdown should be called before reset if needed.
    // Here we assume clean reset or fresh start.
    
    vm->register_count = vm->ctx->register_count;
    vm->registers = MF_ARENA_PUSH(arena, mf_tensor, vm->register_count);

    // Copy initial state from Context Prototypes
    for (u32 i = 0; i < vm->register_count; ++i) {
        mf_tensor* dst = &vm->registers[i];
        const mf_tensor* src = &vm->ctx->tensor_prototypes[i];
        
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
            // Variable: Allocate based on shape
            if (vm->allocator && src->size > 0) {
                 size_t bytes = src->size * mf_dtype_size(src->dtype);
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

static void impl_error(void* impl, int error_code) {
    mf_vm* vm = (mf_vm*)impl;
    vm->error = (mf_vm_error)error_code;
}

void mf_vm_exec(mf_vm* vm) {
    if (!vm->ctx) return;
    
    mf_backend_dispatch_table* backend = vm->ctx->backend;
    
    if (backend && backend->on_map) { 
        // Hook for exec start?
    }

    // Setup Kernel Context
    mf_kernel_ctx kernel_ctx = {
        .impl = vm,
        .map_tensor = (mf_tensor* (*)(void*, u16, mf_access_mode))mf_vm_map_tensor,
        .resize_tensor = (bool (*)(void*, mf_tensor*, const int32_t*, uint8_t))mf_vm_resize_tensor,
        .error = impl_error,
        .batch_size = vm->batch_size
    };
    // Copy Intrinsics
    memcpy(kernel_ctx.global_offset, vm->global_offset, sizeof(vm->global_offset));
    memcpy(kernel_ctx.local_size, vm->local_size, sizeof(vm->local_size));

    for (size_t i = 0; i < vm->ctx->code_count; ++i) {
        if (vm->error != MF_ERROR_NONE) break;

        mf_instruction inst = vm->ctx->code[i];
        if (backend->op_table[inst.opcode]) {
            backend->op_table[inst.opcode](&kernel_ctx, inst.dest_idx, inst.src1_idx, inst.src2_idx);
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
    mf_tensor* t = &vm->registers[idx];
    
    mf_backend_dispatch_table* backend = vm->ctx->backend;
    if (backend && backend->on_map) {
        backend->on_map(vm, t, mode);
    }
    return t;
}

int32_t mf_vm_find_register(mf_vm* vm, const char* name) {
    if (!vm->ctx) return -1;
    for (u32 i = 0; i < vm->ctx->symbol_count; ++i) {
        if (strcmp(vm->ctx->symbols[i].name, name) == 0) {
            return (int32_t)vm->ctx->symbols[i].register_idx;
        }
    }
    return -1;
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
            size_t old_bytes = tensor->capacity_bytes; // Use capacity, not size, for realloc base
            // Wait, realloc expects old_size? 
            // Actually, for Arena we need the VALID data size to copy.
            // But standard realloc doesn't care about data valid size, it cares about block size.
            // BUT, our "Dumb Realloc" does memcpy(ptr, ptr, old_size).
            // So we should pass the VALID DATA SIZE we want to preserve?
            // No, the Allocator contract is usually about MEMORY BLOCK sizes.
            // However, since we don't store block size in Arena, we MUST pass what we want to copy.
            // Let's pass 'tensor->size * type_size' as the 'size to preserve'.
            
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