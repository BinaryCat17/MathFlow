#ifndef MF_VM_H
#define MF_VM_H

#include <mathflow/isa/mf_instruction.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/vm/mf_memory.h>

// Forward decl
typedef struct mf_vm mf_vm;

// --- Enums ---

typedef enum {
    MF_ACCESS_READ = 0,
    MF_ACCESS_WRITE = 1,
    MF_ACCESS_RW = 2
} mf_access_mode;

// --- Backend Interface ---
// Universal Op Function
typedef void (*mf_op_func)(mf_vm* vm, u16 dest, u16 src1, u16 src2);

// Hook for sync (e.g. GPU upload/download)
// Now takes a Tensor pointer
typedef void (*mf_hook_map)(mf_vm* vm, mf_tensor* tensor, mf_access_mode mode);

typedef struct {
    mf_op_func op_table[MF_OP_LIMIT];
    mf_hook_map on_map;
} mf_backend_dispatch_table;

// --- VM State ---
typedef enum {
    MF_ERROR_NONE = 0,
    MF_ERROR_OOM = 1,          // Out of Memory
    MF_ERROR_SHAPE_MISMATCH = 2, // Runtime shape check failed
    MF_ERROR_INVALID_OP = 3    // Unknown opcode
} mf_vm_error;

struct mf_vm {
    // Code & Registers
    mf_instruction* _code;
    size_t _code_count;
    
    mf_tensor* _registers;
    size_t _register_count;

    mf_bin_symbol* _symbols;
    size_t _symbol_count;

    // Execution Context
    mf_backend_dispatch_table* backend;
    
    // Memory Management
    mf_allocator* allocator; // For dynamic tensor data
    
    // State
    mf_vm_error error;
    
    // User Data
    void* user_data;
};

// Helper to load binary from disk
void mf_vm_init(mf_vm* vm, mf_allocator* allocator);
mf_program* mf_vm_load_program_from_file(const char* path, mf_arena* arena);

// Load program into VM memory (allocates registers)
void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena);

// Execute program
void mf_vm_exec(mf_vm* vm);

// Cleanup dynamic memory (tensors allocated by backend)
void mf_vm_shutdown(mf_vm* vm);

// --- Accessors ---
// Returns a pointer to the live tensor in the VM.
// The backend can then read shape/data.
mf_tensor* mf_vm_map_tensor(mf_vm* vm, u16 idx, mf_access_mode mode);

// Named access: Returns register index or -1 if not found.
int32_t mf_vm_find_register(mf_vm* vm, const char* name);

bool mf_vm_resize_tensor(mf_vm* vm, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim);

#endif // MF_VM_H
