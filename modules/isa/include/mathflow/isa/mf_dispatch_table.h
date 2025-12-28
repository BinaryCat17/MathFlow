#ifndef MF_DISPATCH_TABLE_H
#define MF_DISPATCH_TABLE_H

#include <mathflow/isa/mf_kernel_ctx.h>
#include <mathflow/isa/mf_opcodes.h>

// Forward declarations
struct mf_vm;
struct mf_context;

// The Dispatch Table
// Connects the VM Opcodes to the Kernel Implementations.

// Forward declaration
struct mf_vm;

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
 * @param main_vm Pointer to the Main VM (Source of Truth) for reading uniforms/state.
 * @param count_x, count_y Dimensions of the dispatch.
 */
typedef void (*mf_backend_dispatch_func)(
    const struct mf_context* ctx,
    void* pool, 
    const struct mf_vm* main_vm,
    u32 count_x, u32 count_y
);

typedef struct {
    mf_op_func op_table[MF_OP_LIMIT];
    mf_hook_map on_map;
    mf_backend_dispatch_func dispatch;
} mf_backend_dispatch_table;

#endif // MF_DISPATCH_TABLE_H
