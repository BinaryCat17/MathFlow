#include <mathflow/host/mf_host_headless.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/host/mf_loader.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <string.h>

static void print_tensor(const char* name, mf_tensor* t) {
    void* data_ptr = mf_tensor_data(t);
    if (!t || !data_ptr) {
        printf("  %s: (Empty)\n", name ? name : "?");
        return;
    }
    
    printf("  '%s' ", name ? name : "?"); 
    
    printf("Shape: [");
    for(int i=0; i<t->info.ndim; ++i) printf("%d%s", t->info.shape[i], i < t->info.ndim-1 ? "," : "");
    printf("] ");
    
    size_t count = mf_tensor_count(t);
    if (t->info.dtype == MF_DTYPE_F32) {
        printf("F32: {");
        f32* data = (f32*)data_ptr;
        size_t limit = count > 16 ? 16 : count;
        for(size_t i=0; i<limit; ++i) printf("%.2f%s", data[i], i < limit-1 ? ", " : "");
        if (count > limit) printf("... (+%zu)", count - limit);
        printf("}\n");
    } else if (t->info.dtype == MF_DTYPE_I32) {
        printf("I32: {");
        int32_t* data = (int32_t*)data_ptr;
        size_t limit = count > 16 ? 16 : count;
        for(size_t i=0; i<limit; ++i) printf("%d%s", data[i], i < limit-1 ? ", " : "");
        if (count > limit) printf("... (+%zu)", count - limit);
        printf("}\n");
    } else if (t->info.dtype == MF_DTYPE_U8) {
        printf("Bool: {");
        u8* data = (u8*)data_ptr;
        size_t limit = count > 16 ? 16 : count;
        for(size_t i=0; i<limit; ++i) printf("%s%s", data[i] ? "true" : "false", i < limit-1 ? ", " : "");
        printf("}\n");
    }
}

static void debug_print_resource_callback(const char* name, mf_tensor* t, void* user_data) {
    (void)user_data;
    print_tensor(name, t);
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
        // Simple time increment for headless
        mf_host_app_update_system_resources(&app, (f32)f * 0.016f, 0, 0, false, false);

        mf_engine_dispatch(app.engine);
        
        mf_engine_error err = mf_engine_get_error(app.engine);
        if (err != MF_ENGINE_ERR_NONE) {
            MF_LOG_ERROR("Engine error: %d\n", err);
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