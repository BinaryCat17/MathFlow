#ifndef MF_HOST_SDL_H
#define MF_HOST_SDL_H

#include <mathflow/host/mf_host_desc.h>

/**
 * @brief Runs the standard MathFlow Host Loop using SDL2.
 * 
 * This function initializes SDL, creates a window, loads the graph,
 * and runs the loop until the user closes the window.
 * 
 * @param desc Configuration descriptor.
 * @return int 0 on success, non-zero on error.
 */
int mf_host_run(const mf_host_desc* desc);

#endif // MF_HOST_SDL_H