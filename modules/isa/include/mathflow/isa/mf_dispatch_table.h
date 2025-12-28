#ifndef MF_DISPATCH_TABLE_H
#define MF_DISPATCH_TABLE_H

#include <mathflow/isa/mf_kernel_ctx.h>
#include <mathflow/isa/mf_opcodes.h>

// Forward declarations
struct mf_vm;
struct mf_context;

// --- Dispatch Callbacks ---
// Used by the Host to inject data (uniforms) or read back results per-invocation.
typedef void (*mf_vm_job_setup_func)(struct mf_vm* vm, u32 job_idx, void* user_data);
typedef void (*mf_vm_job_finish_func)(struct mf_vm* vm, u32 job_idx, void* user_data);

// --- Backend Interface ---

// Hook for sync (e.g. GPU upload/download)
// Called by the runtime (VM) when a tensor is mapped.
typedef void (*mf_hook_map)(void* impl, mf_tensor* tensor, mf_access_mode mode);

/**
 * @brief Dispatch function for a backend.
 * Handles the execution of a program over a N-dimensional domain.
 * 
 * @param ctx Shared program context.
 * @param pool Optional thread pool (for CPU backend).
 * @param count_x, count_y Dimensions of the dispatch.
 * @param setup_cb Callback before each invocation.
 * @param finish_cb Callback after each invocation.
 * @param user_data Opaque pointer for callbacks.
 */
typedef void (*mf_backend_dispatch_func)(
    const struct mf_context* ctx,
    void* pool, 
    u32 count_x, u32 count_y,
    mf_vm_job_setup_func setup_cb,
    mf_vm_job_finish_func finish_cb,
    void* user_data
);

// The Dispatch Table
// Connects the VM Opcodes to the Kernel Implementations.
typedef struct {
    mf_op_func op_table[MF_OP_LIMIT];
    mf_hook_map on_map;
    mf_backend_dispatch_func dispatch;
} mf_backend_dispatch_table;

#endif // MF_DISPATCH_TABLE_H
