#ifndef MF_HOST_DESC_H
#define MF_HOST_DESC_H

#include <stdbool.h>
#include <mathflow/engine/mf_pipeline.h>

typedef enum {
    MF_ASSET_IMAGE,
    MF_ASSET_FONT
} mf_asset_type;

typedef struct {
    mf_asset_type type;
    const char* resource_name;
    const char* path;
    float font_size; // only for fonts
} mf_host_asset;

// Configuration for the Host Application
typedef struct mf_host_desc {
    const char* window_title;
    int width;
    int height;
    
    // Pipeline configuration (All programs run through this)
    mf_pipeline_desc pipeline;
    bool has_pipeline;

    // Assets to load into resources
    mf_host_asset* assets;
    int asset_count;
    
    // Optional: Number of worker threads (0 = Auto)
    int num_threads;

    // Logging Interval (in seconds) for TRACE logs and screenshots. 0 = Disable periodic logging.
    float log_interval;
    
    // Window Options
    bool fullscreen;
    bool vsync;
    bool resizable;
} mf_host_desc;

/**
 * @brief Initializes the unified logging system for the host application.
 * Creates the 'logs/' directory and sets up both console and file output.
 */
void mf_host_init_logger(void);

/**
 * @brief Cleans up memory allocated within mf_host_desc (e.g. by manifest loader).
 */
void mf_host_desc_cleanup(mf_host_desc* desc);

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

#endif // MF_HOST_DESC_H
