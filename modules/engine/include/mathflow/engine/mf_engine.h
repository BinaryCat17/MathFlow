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

/**
 * @brief Configuration for initializing the engine.
 */
typedef struct mf_engine_desc {
    size_t arena_size;      // Static arena for Code/Metadata (default: 8MB)
    size_t heap_size;       // Dynamic heap for Tensors (default: 64MB)
    mf_backend backend;     // Backend implementation
} mf_engine_desc;

/**
 * @brief Engine status codes.
 */
typedef enum {
    MF_ENGINE_ERR_NONE = 0,
    MF_ENGINE_ERR_OOM,
    MF_ENGINE_ERR_SHAPE,
    MF_ENGINE_ERR_INVALID_OP,
    MF_ENGINE_ERR_RUNTIME
} mf_engine_error;

// --- Lifecycle ---

mf_engine*      mf_engine_create(const mf_engine_desc* desc);
void            mf_engine_destroy(mf_engine* engine);
void            mf_engine_reset(mf_engine* engine);
mf_arena*       mf_engine_get_arena(mf_engine* engine);

// --- Setup ---

/**
 * @brief Binds a pipeline and allocates resources.
 */
void            mf_engine_bind_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe, mf_program** programs);

// --- Execution ---

/**
 * @brief Dispatches the current frame.
 */
void            mf_engine_dispatch(mf_engine* engine);

// --- State & Resource Access ---

/**
 * @brief Returns the current view of a global resource.
 */
mf_tensor*      mf_engine_map_resource(mf_engine* engine, const char* name);

/**
 * @brief Force resize a global resource.
 */
bool            mf_engine_resize_resource(mf_engine* engine, const char* name, const int32_t* new_shape, uint8_t new_ndim);

/**
 * @brief Returns the last error status.
 */
mf_engine_error mf_engine_get_error(mf_engine* engine);

/**
 * @brief Callback for resource iteration.
 */
typedef void (*mf_engine_resource_cb)(const char* name, mf_tensor* tensor, void* user_data);

/**
 * @brief Iterates over all active global resources.
 */
void            mf_engine_iterate_resources(mf_engine* engine, mf_engine_resource_cb cb, void* user_data);

#endif // MF_ENGINE_H