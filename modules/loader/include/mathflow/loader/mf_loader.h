#ifndef MF_LOADER_H
#define MF_LOADER_H

#include <mathflow/base/mf_types.h>
#include <mathflow/isa/mf_backend.h>
#include <mathflow/engine/mf_pipeline.h>
// Forward declarations
typedef struct mf_engine mf_engine;

// --- Backend Management ---

/**
 * @brief Initializes the default backend available in the build.
 * Currently hardcoded to CPU backend.
 * 
 * @param backend Pointer to the backend structure to fill.
 * @param num_threads Number of worker threads (0 = auto).
 */
void mf_loader_init_backend(mf_backend* backend, int num_threads);

// --- Asset Loading ---

/**
 * @brief Loads a graph from file (JSON or Binary) and binds it to the engine.
 * 
 * @param engine The engine instance.
 * @param path Path to the .json or .bin file.
 * @return true if successful.
 */
bool mf_loader_load_graph(mf_engine* engine, const char* path);

/**
 * @brief Loads a complete pipeline description and binds it to the engine.
 * Loads all kernel programs (.bin/.json) referenced in the pipeline.
 * 
 * @param engine The engine instance.
 * @param pipe The pipeline description (parsed from .mfapp).
 * @return true if successful.
 */
bool mf_loader_load_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe);

#endif // MF_LOADER_H