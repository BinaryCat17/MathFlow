#ifndef MF_LOADER_H
#define MF_LOADER_H

#include <mathflow/base/mf_types.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_backend.h>
#include <mathflow/host/mf_host_desc.h>

/**
 * MathFlow Unified Loader
 * 
 * Responsibilities:
 * - Parsing .mfapp manifests
 * - Compiling/Loading kernel programs
 * - Loading assets (Images, Fonts) into Engine resources
 */

// --- Backend Setup ---
void mf_loader_init_backend(mf_backend* backend, int num_threads);

// --- Manifest Parsing ---
int mf_app_load_config(const char* mfapp_path, mf_host_desc* out_desc);

// --- Pipeline Loading ---
bool mf_loader_load_graph(mf_engine* engine, const char* path);
bool            mf_loader_load_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe);
void*           mf_loader_find_section(const char* name, mf_section_type type, size_t* out_size);
bool            mf_loader_load_image(mf_engine* engine, const char* name, const char* path);
bool mf_loader_load_font(mf_engine* engine, const char* resource_name, const char* path, float font_size);

#endif // MF_LOADER_H
