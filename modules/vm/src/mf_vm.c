#include <mathflow/vm/mf_vm.h>
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
    
    // Read descriptors
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + offset);
        offset += sizeof(mf_bin_tensor_desc);
        
        mf_tensor* t = &prog->tensors[i];
        t->dtype = (mf_dtype)desc->dtype;
        t->ndim = desc->ndim;
        memcpy(t->shape, desc->shape, sizeof(i32) * MF_MAX_DIMS);
        t->flags = 0; // Reset flags from binary
        
        t->size = 1;
        for(int k=0; k<t->ndim; ++k) t->size *= t->shape[k];
        if (t->ndim == 0) t->size = 1; // Scalar
    }

    // Read Data Blob
    // Reset offset to iterate again (lazy approach)
    // Actually, we can just calculate pointers.
    // Let's assume binary layout: [Header][Code][Descs][DataBlock]
    // The previous loop advanced 'offset' past all descriptors.
    // So 'offset' is now at the start of DataBlock.
    
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        
        // We need to know if it has data.
        // We can peek back at the descriptor struct in the raw buffer?
        // Or assume: if it's an Input node (constant), it has data.
        // But here we lost the 'is_constant' info. 
        // Let's re-read the descriptor from the buffer to check 'is_constant'.
        
        size_t desc_offset = sizeof(mf_bin_header) + 
                             sizeof(mf_instruction) * head->instruction_count + 
                             sizeof(mf_bin_tensor_desc) * i;
                             
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + desc_offset);
        
        if (desc->is_constant) {
            size_t bytes = mf_dtype_size(t->dtype) * t->size;
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, data + offset, bytes);
            t->data = mem;
            offset += bytes;
        } else {
            t->data = NULL; // Uninitialized
        }
    }

    free(data);
    return prog;
}

void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena) {
    vm->_code = prog->code;
    vm->_code_count = prog->meta.instruction_count;
    vm->_register_count = prog->meta.tensor_count;
    
    // Allocate Registers (Tensors)
    vm->_registers = MF_ARENA_PUSH(arena, mf_tensor, vm->_register_count);
    
    // Copy initial state
    for (u32 i = 0; i < vm->_register_count; ++i) {
        mf_tensor* dst = &vm->_registers[i];
        const mf_tensor* src = &prog->tensors[i];
        
        *dst = *src; // Copy metadata
        dst->flags = 0; // Clear flags (Arena owns memory if any)
        
        if (src->data) {
            // Constant: Deep copy data to new VM memory
            size_t bytes = mf_dtype_size(src->dtype) * src->size;
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, src->data, bytes);
            dst->data = mem;
        } else {
            // Variable: Init empty
            dst->data = NULL;
        }
    }
}

void mf_vm_exec(mf_vm* vm) {
    if (vm->backend && vm->backend->on_map) { 
        // Hook for exec start?
    }

    for (size_t i = 0; i < vm->_code_count; ++i) {
        mf_instruction inst = vm->_code[i];
        if (vm->backend->op_table[inst.opcode]) {
            vm->backend->op_table[inst.opcode](vm, inst.dest_idx, inst.src1_idx, inst.src2_idx);
        }
    }
}

void mf_vm_shutdown(mf_vm* vm) {
    if (!vm->_registers) return;
    
    for (u32 i = 0; i < vm->_register_count; ++i) {
        mf_tensor* t = &vm->_registers[i];
        if (t->data && (t->flags & MF_TENSOR_OWNS_DATA)) {
            free(t->data);
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
