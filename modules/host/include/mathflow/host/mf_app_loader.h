#ifndef MF_APP_LOADER_H
#define MF_APP_LOADER_H

#include <mathflow/host/mf_host.h>

/**
 * @brief Loads an application manifest (.mfapp) and populates the host descriptor.
 * 
 * This function parses the JSON manifest, resolving relative paths (e.g. for the graph entry)
 * against the manifest's location.
 * 
 * @param mfapp_path Path to the .mfapp file.
 * @param out_desc Pointer to the descriptor to populate.
 * @return int 0 on success, non-zero on error.
 */
int mf_app_load_config(const char* mfapp_path, mf_host_desc* out_desc);

#endif // MF_APP_LOADER_H
