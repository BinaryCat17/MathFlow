#ifndef MF_HOST_H
#define MF_HOST_H

#include <stdint.h>
#include <stdbool.h>

// Execution strategy for the runtime
typedef enum mf_host_runtime_type {
    MF_HOST_RUNTIME_SHADER, // Multithreaded, Tiled Rendering (for Visuals)
    MF_HOST_RUNTIME_SCRIPT  // Single-threaded (for Logic/Data)
} mf_host_runtime_type;

// Configuration for the Host Application
typedef struct mf_host_desc {
    const char* window_title;
    int width;
    int height;
    
    // Path to the .json graph file to load on startup
    // (If loaded via .mfapp, this will be the resolved absolute path)
    const char* graph_path;
    
    // Execution Strategy
    mf_host_runtime_type runtime_type;

    // Optional: Number of worker threads (0 = Auto)
    int num_threads;
    
    // Window Options
    bool fullscreen;
    bool vsync;
    bool resizable;
} mf_host_desc;

// Runs the standard MathFlow Host Loop.
// This function initializes SDL, creates a window, loads the graph,
// and runs the loop until the user closes the window.
// Returns 0 on success, non-zero on error.
int mf_host_run(const mf_host_desc* desc);

#endif // MF_HOST_H
