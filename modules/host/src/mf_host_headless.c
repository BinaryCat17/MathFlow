#include <mathflow/host/mf_host_headless.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/loader/mf_loader.h>
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
    if (!desc->graph_path && !desc->has_pipeline) return 1;

    mf_engine_desc engine_desc = {0};
    
    mf_loader_init_backend(&engine_desc.backend, desc->num_threads);

    mf_engine* engine = mf_engine_create(&engine_desc);
    if (!engine) return 1;

    bool loaded = false;
    if (desc->has_pipeline) {
        loaded = mf_loader_load_pipeline(engine, &desc->pipeline);
    } else {
        loaded = mf_loader_load_graph(engine, desc->graph_path);
    }

    if (!loaded) {
        mf_engine_destroy(engine);
        return 1;
    }

    if (desc->width > 0) {
        int w = desc->width;
        int h = (desc->height > 0) ? desc->height : 1;
        
        int32_t shape[3] = { h, w, 4 }; 
        if (h == 1) {
            shape[0] = w;
            mf_engine_resize_resource(engine, "out_Color", shape, 1);
        } else {
            mf_engine_resize_resource(engine, "out_Color", shape, 3);
        }
        
        mf_tensor* t_res = mf_engine_map_resource(engine, "u_Resolution"); 
        if (t_res) {
             void* res_data = mf_tensor_data(t_res);
             if (res_data) {
                 size_t count = mf_tensor_count(t_res);
                 if (count >= 2) {
                     f32* d = (f32*)res_data;
                     d[0] = (f32)w;
                     d[1] = (f32)h;
                 }
             }
        }
        
        mf_tensor* t_rx = mf_engine_map_resource(engine, "u_ResX");
        if (t_rx) {
            void* rx_data = mf_tensor_data(t_rx);
            if (rx_data) *((f32*)rx_data) = (f32)w;
        }
        
        mf_tensor* t_ry = mf_engine_map_resource(engine, "u_ResY");
        if (t_ry) {
            void* ry_data = mf_tensor_data(t_ry);
            if (ry_data) *((f32*)ry_data) = (f32)h;
        }
    }

    MF_LOG_INFO("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        mf_engine_dispatch(engine);
        
        mf_engine_error err = mf_engine_get_error(engine);
        if (err != MF_ENGINE_ERR_NONE) {
            MF_LOG_ERROR("Engine error: %d\n", err);
            break;
        }
        
        // Also check Kernel errors
        // Note: This is simplified, should really iterate all kernels
        
        if (f < 3) {
             MF_LOG_INFO("--- Frame %d ---\
", f);
             mf_engine_iterate_resources(engine, debug_print_resource_callback, NULL);
        }
    }
    
    MF_LOG_INFO("--- Final State ---\
");
    mf_engine_iterate_resources(engine, debug_print_resource_callback, NULL);

    mf_engine_destroy(engine);
    return 0;
}