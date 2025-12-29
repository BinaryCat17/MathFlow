#ifndef MF_VM_H
#define MF_VM_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/base/mf_memory.h>

// Forward decl
typedef struct mf_vm mf_vm;

// --- VM (Execution State / Mutable) ---
// Holds the Heap, Register Values, and Error State.
// NOT Thread-safe: Each thread must have its own VM.
typedef enum {
    MF_ERROR_NONE = 0,
    MF_ERROR_OOM = 1,          // Out of Memory
    MF_ERROR_SHAPE_MISMATCH = 2, // Runtime shape check failed
    MF_ERROR_INVALID_OP = 3    // Unknown opcode
} mf_vm_error;

struct mf_vm {
    // Registers (Active Tensors)
    mf_tensor* registers;
    size_t register_count;

    // Memory Management
    mf_allocator* allocator; // For dynamic tensor data
    
    // Execution Configuration
    u32 batch_size; // Virtual Batching: If > 0, operations process only this many elements. 0 = Full Tensor Size.
    u32 global_offset[3];
    u32 local_size[3];
    u32 global_size[3];

    // State
    mf_vm_error error;
    
    // User Data
    void* user_data;
};

// --- VM API ---
// Initialize a VM instance (allocator can be NULL)
void mf_vm_init(mf_vm* vm, mf_allocator* allocator);

// Load Program Prototypes into VM (Allocates registers based on Program).
// Must be called before usage.
void mf_vm_reset(mf_vm* vm, const mf_program* prog, mf_arena* arena);

// Cleanup dynamic memory (tensors allocated by backend)
void mf_vm_shutdown(mf_vm* vm);

// --- Accessors ---
// Returns a pointer to the live tensor in the VM.
mf_tensor* mf_vm_map_tensor(mf_vm* vm, u16 idx, mf_access_mode mode);

bool mf_vm_resize_tensor(mf_vm* vm, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim);

#endif // MF_VM_H