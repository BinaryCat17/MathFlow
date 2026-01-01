#ifndef MF_HOST_INTERNAL_H
#define MF_HOST_INTERNAL_H

#include <mathflow/host/mf_host_desc.h>
#include <mathflow/engine/mf_engine.h>

/**
 * @brief Shared context for a running MathFlow application.
 * Internal to the host module.
 */
typedef struct {
    mf_host_desc desc;
    mf_engine* engine;
    
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
 * @brief Initializes the host application context.
 */
int mf_host_app_init(mf_host_app* app, const mf_host_desc* desc);

/**
 * @brief Sets simulation time.
 */
void mf_host_app_set_time(mf_host_app* app, float current_time);

/**
 * @brief Sets output resolution.
 */
void mf_host_app_set_resolution(mf_host_app* app, int width, int height);

/**
 * @brief Sets mouse state (GUI specific).
 */
void mf_host_app_set_mouse(mf_host_app* app, float x, float y, bool lmb, bool rmb);

/**
 * @brief Shuts down the application context.
 */
void mf_host_app_cleanup(mf_host_app* app);

#endif // MF_HOST_INTERNAL_H
