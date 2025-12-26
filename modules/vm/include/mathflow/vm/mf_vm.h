#ifndef MF_VM_H
#define MF_VM_H

#include <mathflow/isa/mf_instruction.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/vm/mf_memory.h>

// Forward decl
struct mf_vm_t;

// --- Enums ---

typedef enum {
    MF_ACCESS_READ = 0,
    MF_ACCESS_WRITE = 1,
    MF_ACCESS_RW = 2
} mf_access_mode;

// --- Backend Interface ---
// Universal Op Function
typedef void (*mf_op_func)(struct mf_vm_t* vm, u16 dest, u16 src1, u16 src2);

// Hook for sync (e.g. GPU upload/download)
// Now takes a Tensor pointer
typedef void (*mf_hook_map)(struct mf_vm_t* vm, mf_tensor* tensor, mf_access_mode mode);

typedef struct {
    mf_op_func op_table[MF_OP_COUNT];
    mf_hook_map on_map;
} mf_backend_dispatch_table;

// --- VM State ---
typedef struct mf_vm_t {
    const mf_backend_dispatch_table* backend;

    mf_instruction* _code;
    size_t _code_count;
    
    // The Tensor Register File
    // This is an array of tensors.
    // The *data pointers* inside these tensors point to allocated memory.
    mf_tensor* _registers; 
    u32 _register_count;
} mf_vm;

// Load a program into the VM
void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena);

// Load a program from file
mf_program* mf_vm_load_program_from_file(const char* path, mf_arena* arena);

// Execute
void mf_vm_exec(mf_vm* vm);

// --- Accessor API ---
// Returns a pointer to the live tensor in the VM.
// The backend can then read shape/data.
mf_tensor* mf_vm_map_tensor(mf_vm* vm, u16 idx, mf_access_mode mode);

#endif // MF_VM_H