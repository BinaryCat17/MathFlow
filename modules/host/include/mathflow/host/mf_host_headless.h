#ifndef MF_HOST_HEADLESS_H
#define MF_HOST_HEADLESS_H

#include <mathflow/engine/mf_engine.h>

/**
 * @brief Runs the engine in headless mode (CLI).
 * Executes the graph for a specified number of frames and prints output tensors.
 * 
 * @param engine The initialized engine with a bound program.
 * @param frames Number of frames to simulate.
 * @return int Exit code (0 on success).
 */
int mf_host_run_headless(mf_engine* engine, int frames);

#endif // MF_HOST_HEADLESS_H
