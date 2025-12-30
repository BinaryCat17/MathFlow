#ifndef MF_ENGINE_H
#define MF_ENGINE_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_backend.h>
#include <mathflow/base/mf_types.h>
#include <mathflow/engine/mf_pipeline.h>

// Forward declarations
typedef struct mf_program mf_program;
typedef struct mf_arena mf_arena;

// Opaque Engine Handle
typedef struct mf_engine mf_engine;

// Configuration for initializing the engine
typedef struct mf_engine_desc {
    // Size of the static arena for Program Code & Metadata.
    // Default: 8MB if set to 0.
    size_t arena_size;

    // Size of the dynamic heap for Variables (State).
    // Default: 64MB if set to 0.
    size_t heap_size;

    // Backend Implementation (Required)
    // The engine copies this table, so it can be on stack.
    mf_backend backend;
} mf_engine_desc;

// --- Lifecycle ---

// Creates a new Engine instance.
// Allocates internal state (VM, Heap, Arena).
mf_engine* mf_engine_create(const mf_engine_desc* desc);

// Destroys the engine and frees all resources.
void mf_engine_destroy(mf_engine* engine);

// Resets the engine state (Heap, Arena, Program), keeping resources (Threads, Buffers) alive.
// Useful for hot-reloading graphs.
void mf_engine_reset(mf_engine* engine);

// Returns the internal arena used for program allocation.
// Required for external loaders/compilers.
mf_arena* mf_engine_get_arena(mf_engine* engine);

// --- Setup ---

// Binds a pipeline configuration to the engine.
// Allocates global resources and initializes all kernels.
// @param programs Array of mf_program* matching pipe->kernel_count order.
void mf_engine_bind_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe, mf_program** programs);

// --- Execution ---

/**
 * @brief Dispatches the current pipeline.
 * Iterates over all kernels, determines their domain automatically based on Output resources,
 * and executes them using the backend.
 */
void mf_engine_dispatch(mf_engine* engine);

// --- State Access (Single Source of Truth) ---

// Maps a global resource by name.
// Returns a tensor descriptor pointing to the current buffer (Ping or Pong).
mf_tensor* mf_engine_map_resource(mf_engine* engine, const char* name);

// Resizes a global resource buffer (e.g. for window resize).
bool mf_engine_resize_resource(mf_engine* engine, const char* name, const int32_t* new_shape, uint8_t new_ndim);

typedef enum {
    MF_ENGINE_ERR_NONE = 0,
    MF_ENGINE_ERR_OOM,
    MF_ENGINE_ERR_SHAPE,
    MF_ENGINE_ERR_INVALID_OP
} mf_engine_error;

mf_engine_error mf_engine_get_error(mf_engine* engine);

typedef void (*mf_engine_resource_cb)(const char* name, mf_tensor* tensor, void* user_data);

// Iterates over all global resources in the Pipeline.
void mf_engine_iterate_resources(mf_engine* engine, mf_engine_resource_cb cb, void* user_data);

#endif // MF_ENGINE_H
