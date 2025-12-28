#ifndef MF_HOST_H
#define MF_HOST_H

#include <stdint.h>
#include <stdbool.h>

// Configuration for the Host Application
typedef struct mf_host_desc {
    const char* window_title;
    int width;
    int height;
    
    // Path to the .json graph file to load on startup
    const char* graph_path;
    
    // Optional: Number of worker threads (0 = Auto)
    int num_threads;
    
    // Optional: Fullscreen mode
    bool fullscreen;
} mf_host_desc;

// Runs the standard MathFlow Host Loop.
// This function initializes SDL, creates a window, loads the graph,
// and runs the loop until the user closes the window.
// Returns 0 on success, non-zero on error.
int mf_host_run(const mf_host_desc* desc);

#endif // MF_HOST_H
