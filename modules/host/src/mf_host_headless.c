#include <mathflow/host/mf_host_headless.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/loader/mf_loader.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <string.h>

static void print_tensor(const char* name, mf_tensor* t) {
    if (!t || !t->data) {
        printf("  %s: (Empty)\n", name ? name : "?");
        return;
    }
    
    printf("  '%s' ", name ? name : "?"); 
    
    printf("Shape: [");
    for(int i=0; i<t->ndim; ++i) printf("%d%s", t->shape[i], i < t->ndim-1 ? "," : "");
    printf("] ");
    
    if (t->dtype == MF_DTYPE_F32) {
        printf("F32: {");
        f32* data = (f32*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) printf("%.2f%s", data[i], i < limit-1 ? ", " : "");
        if (t->size > limit) printf("... (+%zu)", t->size - limit);
        printf("}\n");
    } else if (t->dtype == MF_DTYPE_I32) {
        printf("I32: {");
        int32_t* data = (int32_t*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) printf("%d%s", data[i], i < limit-1 ? ", " : "");
        if (t->size > limit) printf("... (+%zu)", t->size - limit);
        printf("}\n");
    } else if (t->dtype == MF_DTYPE_U8) {
        printf("Bool: {");
        u8* data = (u8*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) printf("%s%s", data[i] ? "true" : "false", i < limit-1 ? ", " : "");
        printf("}\n");
    }
}

static void debug_print_resource_callback(const char* name, mf_tensor* t, void* user_data) {
    print_tensor(name, t);
}

int mf_host_run_headless(const mf_host_desc* desc, int frames) {
    if (!desc) return 1;
    if (!desc->graph_path && !desc->has_pipeline) return 1;

    mf_engine_desc engine_desc = {0};
    
    // Init Backend via Loader (Injection)
    mf_loader_init_backend(&engine_desc.backend, desc->num_threads);

    mf_engine* engine = mf_engine_create(&engine_desc);
    if (!engine) return 1;

    bool loaded = false;
    if (desc->has_pipeline) {
        loaded = mf_loader_load_pipeline(engine, &desc->pipeline);
    } else {
        // Now this automatically synthesizes a pipeline internally
        loaded = mf_loader_load_graph(engine, desc->graph_path);
    }

    if (!loaded) {
        mf_engine_destroy(engine);
        return 1;
    }

    MF_LOG_INFO("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        // Dispatch always using pipeline logic
        // For implicit single-graph pipelines, we use the requested size
        mf_engine_dispatch(engine, desc->width > 0 ? desc->width : 1, desc->height > 0 ? desc->height : 1);
        
        mf_engine_error err = mf_engine_get_error(engine);
        if (err != MF_ENGINE_ERR_NONE) {
            MF_LOG_ERROR("Runtime error %d\n", err);
            break;
        }
        
        if (f < 3) {
             MF_LOG_INFO("--- Frame %d ---\n", f);
             mf_engine_iterate_resources(engine, debug_print_resource_callback, NULL);
        }
    }
    
    MF_LOG_INFO("--- Final State ---\n");
    mf_engine_iterate_resources(engine, debug_print_resource_callback, NULL);

    mf_engine_destroy(engine);
    return 0;
}