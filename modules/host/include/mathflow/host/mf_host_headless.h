#ifndef MF_HOST_HEADLESS_H
#define MF_HOST_HEADLESS_H

#include <mathflow/host/mf_host_desc.h>

/**
 * @brief Runs the engine in headless mode (CLI).
 * Initializes the engine, loads the graph specified in the descriptor,
 * executes for a specified number of frames, and prints output.
 * 
 * @param desc Configuration descriptor (graph path, settings).
 * @param frames Number of frames to simulate.
 * @return int Exit code (0 on success).
 */
int mf_host_run_headless(const mf_host_desc* desc, int frames);

#endif // MF_HOST_HEADLESS_H
