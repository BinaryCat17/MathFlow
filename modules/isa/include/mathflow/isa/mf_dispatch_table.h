#ifndef MF_DISPATCH_TABLE_H
#define MF_DISPATCH_TABLE_H

#include <mathflow/isa/mf_kernel_ctx.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_state.h>

// Forward declarations
struct mf_exec_ctx;
struct mf_program;

// The Dispatch Table
// Connects the VM Opcodes to the Kernel Implementations.

// --- Backend Interface ---

// Hook for sync (e.g. GPU upload/download)
// Called by the runtime when a tensor is mapped.
typedef void (*mf_hook_map)(void* impl, mf_tensor* tensor, mf_access_mode mode);

/**
 * @brief Dispatch function for a backend.
 * Handles the execution of a program over a N-dimensional domain.
 * 
 * @param backend_state Internal state of the backend (e.g. thread pool).
 * @param program The program to execute (Code/Prototypes).
 * @param state Pointer to the persistent state (Registers/Data).
 * @param count_x, count_y Dimensions of the dispatch.
 */
typedef void (*mf_backend_dispatch_func)(
    void* backend_state,
    const struct mf_program* program,
    mf_state* state,
    u32 count_x, u32 count_y
);

// Cleanup function for backend resources
typedef void (*mf_backend_shutdown_func)(void* backend_state);

typedef struct {
    // Internal Backend State (Opaque to Engine)
    void* state;

    mf_op_func op_table[MF_OP_LIMIT];
    mf_hook_map on_map;
    
    mf_backend_dispatch_func dispatch;
    mf_backend_shutdown_func shutdown;
} mf_backend_dispatch_table;

#endif // MF_DISPATCH_TABLE_H