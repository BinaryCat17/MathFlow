#ifndef MF_BACKEND_H
#define MF_BACKEND_H

#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_state.h>

// Forward declarations
struct mf_exec_ctx;
struct mf_program;

/**
 * @brief Backend Interface.
 * Handles the execution of a program over a N-dimensional domain.
 */

// Hook for sync (e.g. GPU upload/download)
// Called by the runtime when a tensor is mapped.
typedef void (*mf_hook_map)(void* impl, mf_tensor* tensor, mf_access_mode mode);

/**
 * @brief Dispatch function for a backend.
 */
typedef void (*mf_backend_dispatch_func)(
    void* backend_state,
    const struct mf_program* program,
    mf_state* state,
    const mf_tensor* domain
);

// Cleanup function for backend resources
typedef void (*mf_backend_shutdown_func)(void* backend_state);

typedef struct {
    // Internal Backend State (Opaque to Engine)
    void* state;

    mf_hook_map on_map;
    
    mf_backend_dispatch_func dispatch;
    mf_backend_shutdown_func shutdown;
} mf_backend;

#endif // MF_BACKEND_H
