#ifndef MF_ASSET_LOADER_H
#define MF_ASSET_LOADER_H

#include <mathflow/engine/mf_engine.h>
#include <stdbool.h>

/**
 * @brief Loads a MathFlow graph from a file into the Engine.
 * 
 * Supports:
 * - .json: Compiles source graph (uses mf_compiler).
 * - .bin:  Loads binary program directly.
 * 
 * The program data is allocated in the Engine's Arena.
 * 
 * @param engine The engine instance (must be initialized).
 * @param path Path to the file.
 * @return true on success.
 */
bool mf_asset_loader_load(mf_engine* engine, const char* path);

#endif // MF_ASSET_LOADER_H
