#include <mathflow/host/mf_host_headless.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/base/mf_log.h>
#include "mf_host_internal.h"
#include "mf_loader.h"
#include <stdio.h>
#include <string.h>

static void debug_print_resource_callback(const char* name, mf_tensor* t, void* user_data) {
    (void)user_data;
    mf_tensor_print(name, t);
}

int mf_host_run_headless(const mf_host_desc* desc, int frames) {
    if (!desc) return 1;

    mf_host_app app;
    if (mf_host_app_init(&app, desc) != 0) {
        MF_LOG_ERROR("Failed to initialize Host App");
        return 1;
    }

    MF_LOG_INFO("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        mf_host_app_set_time(&app, (f32)f * 0.016f);

        mf_engine_error err = mf_host_app_step(&app);
        if (err != MF_ENGINE_ERR_NONE) {
            MF_LOG_ERROR("Engine failure: %s", mf_engine_error_to_str(err));
            break;
        }
        
        if (f < 3) {
             MF_LOG_INFO("--- Frame %d ---\n", f);
             mf_engine_iterate_resources(app.engine, debug_print_resource_callback, NULL);
        }
    }
    
    MF_LOG_INFO("--- Final State ---\n");
    mf_engine_iterate_resources(app.engine, debug_print_resource_callback, NULL);

    mf_host_app_cleanup(&app);
    return 0;
}