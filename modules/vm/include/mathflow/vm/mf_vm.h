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

// --- Parallel Execution API ---

typedef void (*mf_vm_job_setup_func)(mf_vm* vm, u32 job_idx, void* user_data);
typedef void (*mf_vm_job_finish_func)(mf_vm* vm, u32 job_idx, void* user_data);

// Worker lifecycle (used by mf_engine to create the thread pool)
void* mf_vm_worker_init(int thread_idx, void* user_data);
void mf_vm_worker_cleanup(void* thread_local_data, void* user_data);

/**
 * @brief Executes a program in parallel over a range of jobs.
 * This is a high-level orchestration function that uses a thread pool.
 * It creates per-thread VMs and heaps internally.
 * 
 * @param ctx The shared program context.
 * @param pool The thread pool to use.
 * @param job_count Total number of parallel jobs.
 * @param setup_cb Called on each thread's VM before execution of a job.
 * @param finish_cb Called on each thread's VM after execution of a job.
 * @param user_data Passed to callbacks.
 */
void mf_vm_exec_parallel(
    const mf_context* ctx,
    mf_thread_pool* pool,
    u32 job_count,
    mf_vm_job_setup_func setup_cb,
    mf_vm_job_finish_func finish_cb,
    void* user_data
);

// --- Accessors ---
// Returns a pointer to the live tensor in the VM.
mf_tensor* mf_vm_map_tensor(mf_vm* vm, u16 idx, mf_access_mode mode);

// Named access: Returns register index or -1 if not found.
// Lookups are done via the Context's Symbol Table.
int32_t mf_vm_find_register(mf_vm* vm, const char* name);

bool mf_vm_resize_tensor(mf_vm* vm, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim);

// Helper: Load program from file and return it (uses Arena)
mf_program* mf_vm_load_program_from_file(const char* path, mf_arena* arena);

#endif // MF_VM_H
