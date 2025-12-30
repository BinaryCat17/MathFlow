#ifndef MF_HOST_DESC_H
#define MF_HOST_DESC_H

#include <stdbool.h>
#include <mathflow/engine/mf_pipeline.h>

// Configuration for the Host Application
typedef struct mf_host_desc {
    const char* window_title;
    int width;
    int height;
    
    // Path to the .json graph file to load on startup (Legacy Single-Graph mode)
    const char* graph_path;
    
    // Pipeline configuration (New Multi-Kernel mode)
    mf_pipeline_desc pipeline;
    bool has_pipeline;
    
    // Optional: Number of worker threads (0 = Auto)
    int num_threads;
    
    // Window Options
    bool fullscreen;
    bool vsync;
    bool resizable;
} mf_host_desc;

#endif // MF_HOST_DESC_H
