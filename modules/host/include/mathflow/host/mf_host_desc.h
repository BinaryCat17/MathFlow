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
 * @brief Cleans up memory allocated within mf_host_desc (e.g. by manifest loader).
 */
void mf_host_desc_cleanup(mf_host_desc* desc);

// --- Host Context (Application Lifecycle) ---

typedef struct mf_engine mf_engine;

/**
 * @brief Shared context for a running MathFlow application.
 */
typedef struct {
    mf_host_desc desc;
    mf_engine* engine;
    
    // Cached standard resource handles (optional but efficient)
    struct {
        mf_tensor* time;
        mf_tensor* mouse;
        mf_tensor* resolution;
        mf_tensor* res_x;
        mf_tensor* res_y;
        mf_tensor* aspect;
    } resources;

    bool is_initialized;
} mf_host_app;

/**
 * @brief Initializes the host application using the provided descriptor.
 * This creates the engine, loads the pipeline, and binds standard resources.
 */
int mf_host_app_init(mf_host_app* app, const mf_host_desc* desc);

/**
 * @brief Sets the global time resource (u_Time).
 */
void mf_host_app_set_time(mf_host_app* app, float current_time);

/**
 * @brief Sets the output resolution and updates associated resources (out_Color, u_Resolution, etc).
 */
void mf_host_app_set_resolution(mf_host_app* app, int width, int height);

/**
 * @brief Sets mouse input resources (u_Mouse, u_MouseX, u_MouseY).
 */
void mf_host_app_set_mouse(mf_host_app* app, float x, float y, bool lmb, bool rmb);

/**
 * @brief Shuts down the application and frees all resources.
 */
void mf_host_app_cleanup(mf_host_app* app);

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
