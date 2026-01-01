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
 * This creates the engine, loads the pipeline/graph, and binds standard resources.
 */
int mf_host_app_init(mf_host_app* app, const mf_host_desc* desc);

/**
 * @brief Updates standard resources (time, resolution, mouse) in the context.
 */
void mf_host_app_update_system_resources(mf_host_app* app, float delta_time, float mouse_x, float mouse_y, bool lmb, bool rmb);

/**
 * @brief Handles window resize events, updating out_Color and resolution resources.
 */
void mf_host_app_handle_resize(mf_host_app* app, int width, int height);

/**
 * @brief Shuts down the application and frees all resources.
 */
void mf_host_app_cleanup(mf_host_app* app);

#endif // MF_HOST_DESC_H
