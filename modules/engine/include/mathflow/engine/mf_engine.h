#ifndef MF_ENGINE_H
#define MF_ENGINE_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/base/mf_types.h>

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

    // Number of worker threads for parallel execution.
    // Default: 0 (Auto-detect CPU count).
    int num_threads;
} mf_engine_desc;

// --- Lifecycle ---

// Creates a new Engine instance.
// Allocates internal state (VM, Heap, Arena).
mf_engine* mf_engine_create(const mf_engine_desc* desc);

// Destroys the engine and frees all resources.
void mf_engine_destroy(mf_engine* engine);

// Returns the internal arena used for program allocation.
// Required for external loaders/compilers.
mf_arena* mf_engine_get_arena(mf_engine* engine);

// --- Setup ---

// Binds a loaded program to the engine.
// Initializes the Execution Context and allocates registers in the Heap.
// @param prog Pointer to program data (must be valid for engine lifetime).
void mf_engine_bind_program(mf_engine* engine, mf_program* prog);

// --- Execution ---

/**
 * @brief Dispatches the current program over a 2D domain.
 * Automatically uses the active backend and thread pool.
 * 
 * If count_x=1 and count_y=1, it runs as a single task (Script Mode).
 * Otherwise, it runs in parallel (Shader Mode), propagating state from the Main VM.
 */
void mf_engine_dispatch(
    mf_engine* engine, 
    u32 count_x, u32 count_y
);

// --- State Access (Single Source of Truth) ---

// Finds a register index by name. Returns -1 if not found.
int32_t mf_engine_find_register(mf_engine* engine, const char* name);

// Maps a tensor from the Engine's main state.
mf_tensor* mf_engine_map_tensor(mf_engine* engine, u16 reg_idx, mf_access_mode mode);

// Resizes a tensor in the Engine's main state.
bool mf_engine_resize_tensor(mf_engine* engine, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim);

typedef enum {
    MF_ENGINE_ERR_NONE = 0,
    MF_ENGINE_ERR_OOM,
    MF_ENGINE_ERR_SHAPE,
    MF_ENGINE_ERR_INVALID_OP
} mf_engine_error;

mf_engine_error mf_engine_get_error(mf_engine* engine);

typedef void (*mf_engine_register_cb)(u16 reg_idx, const char* name, mf_tensor* tensor, void* user_data);

// Iterates over all registers in the Engine's main state.
void mf_engine_iterate_registers(mf_engine* engine, mf_engine_register_cb cb, void* user_data);

#endif // MF_ENGINE_H