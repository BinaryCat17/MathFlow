#include <mathflow/vm/mf_vm.h>
#include <mathflow/vm/mf_memory.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static char* read_file_bin(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(length);
    if (!buffer) return NULL;
    fread(buffer, 1, length, f);
    fclose(f);
    if (out_len) *out_len = length;
    return buffer;
}

void mf_vm_init(mf_vm* vm, mf_allocator* allocator) {
    memset(vm, 0, sizeof(mf_vm));
    vm->allocator = allocator;
}

mf_program* mf_vm_load_program_from_file(const char* path, mf_arena* arena) {
    size_t len = 0;
    char* data = read_file_bin(path, &len);
    if (!data) return NULL;

    if (len < sizeof(mf_bin_header)) { free(data); return NULL; }
    
    mf_bin_header* head = (mf_bin_header*)data;
    if (head->magic != MF_BINARY_MAGIC || head->version != MF_BINARY_VERSION) {
        printf("Invalid binary version.\n");
        free(data);
        return NULL;
    }

    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta = *head;

    // Calc offsets
    size_t offset = sizeof(mf_bin_header);
    
    // Code
    prog->code = MF_ARENA_PUSH(arena, mf_instruction, head->instruction_count);
    memcpy(prog->code, data + offset, sizeof(mf_instruction) * head->instruction_count);
    offset += sizeof(mf_instruction) * head->instruction_count;

    // Tensor Descriptors
    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, head->tensor_count);
    
    // Read descriptors (First pass to set up structs)
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + offset);
        offset += sizeof(mf_bin_tensor_desc);
        
        mf_tensor* t = &prog->tensors[i];
        t->dtype = (mf_dtype)desc->dtype;
        t->ndim = desc->ndim;
        memcpy(t->shape, desc->shape, sizeof(i32) * MF_MAX_DIMS);
        t->flags = 0; 
        
        t->size = 1;
        for(int k=0; k<t->ndim; ++k) t->size *= t->shape[k];
        if (t->ndim == 0) t->size = 1; // Scalar
        
        t->capacity_bytes = 0; // Program descriptors don't own memory yet
    }

    // Read Data Blob
    // Reset offset to iterate again to find constants data
    size_t desc_start_offset = sizeof(mf_bin_header) + sizeof(mf_instruction) * head->instruction_count;
    
    // Data starts after all descriptors
    size_t data_start_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * head->tensor_count;
    offset = data_start_offset;

    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        
        // Re-read descriptor to check is_constant
        size_t this_desc_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * i;
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + this_desc_offset);
        
        if (desc->is_constant) {
            size_t bytes = mf_dtype_size(t->dtype) * t->size;
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, data + offset, bytes);
            t->data = mem;
            t->capacity_bytes = bytes; 
            // Note: Constants in program struct point to Arena memory.
            offset += bytes;
        } else {
            t->data = NULL;
            t->capacity_bytes = 0;
        }
    }

    free(data);
    return prog;
}

void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena) {
    vm->_code = prog->code;
    vm->_code_count = prog->meta.instruction_count;
    vm->_register_count = prog->meta.tensor_count;
    
    // Allocate Registers (Tensors) in Arena (Metadata only)
    vm->_registers = MF_ARENA_PUSH(arena, mf_tensor, vm->_register_count);
    
    // Copy initial state
    for (u32 i = 0; i < vm->_register_count; ++i) {
        mf_tensor* dst = &vm->_registers[i];
        const mf_tensor* src = &prog->tensors[i];
        
        *dst = *src; // Copy metadata
        dst->flags = 0; 
        
        if (src->data) {
            // Constant: Deep copy data to Allocator memory?
            // Or keep in Arena if read-only?
            // To be safe and unified, let's alloc in VM allocator.
            // THIS IS KEY FOR DYNAMIC EXECUTION: Everything in VM should be manageable.
            
            if (vm->allocator) {
                size_t bytes = src->capacity_bytes;
                dst->data = vm->allocator->alloc(vm->allocator, bytes);
                memcpy(dst->data, src->data, bytes);
                dst->capacity_bytes = bytes;
                dst->flags |= MF_TENSOR_OWNS_DATA | MF_TENSOR_DYNAMIC;
            } else {
                 // Fallback if no allocator (legacy mode), point to arena
                 dst->data = src->data; // Read-only ref
                 // Do not set OWNS_DATA so we don't try to free it
            }
        } else {
            // Variable: Init empty
            dst->data = NULL;
            dst->capacity_bytes = 0;
            dst->flags |= MF_TENSOR_DYNAMIC; // Variables are dynamic
        }
    }
}

void mf_vm_exec(mf_vm* vm) {
    if (vm->backend && vm->backend->on_map) { 
        // Hook for exec start?
    }

    for (size_t i = 0; i < vm->_code_count; ++i) {
        if (vm->error != MF_ERROR_NONE) break;

        mf_instruction inst = vm->_code[i];
        if (vm->backend->op_table[inst.opcode]) {
            vm->backend->op_table[inst.opcode](vm, inst.dest_idx, inst.src1_idx, inst.src2_idx);
        }
    }
}

void mf_vm_shutdown(mf_vm* vm) {
    if (!vm->_registers || !vm->allocator) return;
    
    for (u32 i = 0; i < vm->_register_count; ++i) {
        mf_tensor* t = &vm->_registers[i];
        if (t->data && (t->flags & MF_TENSOR_OWNS_DATA)) {
            vm->allocator->free(vm->allocator, t->data);
            t->data = NULL;
            t->flags &= ~MF_TENSOR_OWNS_DATA;
        }
    }
}

mf_tensor* mf_vm_map_tensor(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (idx >= vm->_register_count) return NULL;
    mf_tensor* t = &vm->_registers[idx];
    
    if (vm->backend && vm->backend->on_map) {
        vm->backend->on_map(vm, t, mode);
    }
    return t;
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
    memcpy(tensor->shape, new_shape, sizeof(int32_t) * MF_MAX_DIMS); // Copy up to MAX, safe enough
    tensor->size = new_count;
    
    // Check capacity
    if (needed_bytes > tensor->capacity_bytes) {
        // Reallocate
        // Use geometric growth? For now exact fit to save memory on embedded-like targets
        // Or +20%? Let's do exact fit for now.
        
        void* new_ptr = NULL;
        if (tensor->data && (tensor->flags & MF_TENSOR_OWNS_DATA)) {
            new_ptr = vm->allocator->realloc(vm->allocator, tensor->data, needed_bytes);
        } else {
            // First alloc or transitioning from static (arena) to dynamic
            new_ptr = vm->allocator->alloc(vm->allocator, needed_bytes);
            if (tensor->data && !(tensor->flags & MF_TENSOR_OWNS_DATA)) {
                // Should we copy old data? 
                // Resize usually implies "I want to write new data here", so maybe not?
                // But realloc preserves data.
                // Let's assume if we resize, we might want to preserve.
                size_t old_bytes = tensor->size * type_size; // Caution: old size might be smaller or larger
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