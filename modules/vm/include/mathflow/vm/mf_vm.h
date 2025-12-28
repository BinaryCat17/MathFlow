#ifndef MF_VM_H
#define MF_VM_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_kernel_ctx.h>
#include <mathflow/isa/mf_dispatch_table.h>
#include <mathflow/base/mf_memory.h>
#include <mathflow/base/mf_thread_pool.h>

// Forward decl
typedef struct mf_vm mf_vm;
typedef struct mf_context mf_context;


// --- Context (Immutable / Shared) ---
// Holds the Program Code, Symbols, and Backend Interface.
// Thread-safe: Can be shared across multiple VMs.
struct mf_context {
    // Code
    mf_instruction* code;
    size_t code_count;
    
    // Symbols
    mf_bin_symbol* symbols;
    size_t symbol_count;

    // Initial Tensor State (Prototypes from Program)
    mf_tensor* tensor_prototypes;
    size_t register_count;

    // Execution Logic
    mf_backend_dispatch_table* backend;
};

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
    const mf_context* ctx; // Shared Context
    
    // Registers (Active Tensors)
    mf_tensor* registers;
    size_t register_count;

    // Memory Management
    mf_allocator* allocator; // For dynamic tensor data
    
    // Execution Configuration
    u32 batch_size; // Virtual Batching: If > 0, operations process only this many elements. 0 = Full Tensor Size.
    u32 global_offset[3];
    u32 local_size[3];

    // State
    mf_vm_error error;
    
    // User Data
    void* user_data;
};

// --- Context API ---
// Initializes a context with a program and backend.
// The Context does NOT own the Program memory (it just points to it).
void mf_context_init(mf_context* ctx, const mf_program* prog, mf_backend_dispatch_table* backend);

// --- VM API ---
// Initialize a VM instance attached to a Context.
void mf_vm_init(mf_vm* vm, const mf_context* ctx, mf_allocator* allocator);

// Load Program Data into VM (Allocates registers based on Context prototypes).
// Must be called before exec.
void mf_vm_reset(mf_vm* vm, mf_arena* arena);

// Execute program
void mf_vm_exec(mf_vm* vm);

// Cleanup dynamic memory (tensors allocated by backend)
void mf_vm_shutdown(mf_vm* vm);

// --- Accessors ---
// Returns a pointer to the live tensor in the VM.
mf_tensor* mf_vm_map_tensor(mf_vm* vm, u16 idx, mf_access_mode mode);

// Named access: Returns register index or -1 if not found.
// Lookups are done via the Context's Symbol Table.
int32_t mf_vm_find_register(mf_vm* vm, const char* name);

bool mf_vm_resize_tensor(mf_vm* vm, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim);

#endif // MF_VM_H
