#include <mathflow/host/mf_host_desc.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/base/mf_log.h>
#include "mf_host_internal.h"
#include "mf_loader.h"
#include <stdlib.h>
#include <string.h>

void mf_host_desc_cleanup(mf_host_desc* desc) {
    if (!desc) return;

    if (desc->window_title) free((void*)desc->window_title);

    // Cleanup Pipeline
    if (desc->has_pipeline) {
        for (u32 i = 0; i < desc->pipeline.resource_count; ++i) {
            free((void*)desc->pipeline.resources[i].name);
        }
        free(desc->pipeline.resources);

        for (u32 i = 0; i < desc->pipeline.kernel_count; ++i) {
            mf_pipeline_kernel* k = &desc->pipeline.kernels[i];
            free((void*)k->id);
            free((void*)k->graph_path);
            for (u32 b = 0; b < k->binding_count; ++b) {
                free((void*)k->bindings[b].kernel_port);
                free((void*)k->bindings[b].global_resource);
            }
            free(k->bindings);
        }
        free(desc->pipeline.kernels);
    }

    // Cleanup Assets
    for (int i = 0; i < desc->asset_count; ++i) {
        free((void*)desc->assets[i].resource_name);
        free((void*)desc->assets[i].path);
    }
    free(desc->assets);

    memset(desc, 0, sizeof(mf_host_desc));
}

int mf_host_app_init(mf_host_app* app, const mf_host_desc* desc) {
    if (!app || !desc) return -1;
    memset(app, 0, sizeof(mf_host_app));
    
    // Deep copy of descriptor is not strictly necessary if we assume ownership
    // But for safety, we just keep the reference or copy it.
    // Given mf_host_desc_cleanup frees strings, we should probably own it.
    app->desc = *desc; 

    mf_engine_desc engine_desc = {0};
    engine_desc.arena_size = 32 * 1024 * 1024; 
    engine_desc.heap_size = 128 * 1024 * 1024; 
    mf_loader_init_backend(&engine_desc.backend, desc->num_threads);

    app->engine = mf_engine_create(&engine_desc);
    if (!app->engine) return -2;

    bool loaded = false;
    if (desc->has_pipeline) {
        loaded = mf_loader_load_pipeline(app->engine, &desc->pipeline);
    }

    if (!loaded) {
        mf_engine_destroy(app->engine);
        app->engine = NULL;
        return -3;
    }

    // Load Assets
    for (int i = 0; i < desc->asset_count; ++i) {
        mf_host_asset* asset = &desc->assets[i];
        if (asset->type == MF_ASSET_IMAGE) {
            mf_loader_load_image(app->engine, asset->resource_name, asset->path);
        } else if (asset->type == MF_ASSET_FONT) {
            mf_loader_load_font(app->engine, asset->resource_name, asset->path, asset->font_size);
        }
    }

    // Initial resolution setup for output and uniforms
    mf_host_app_set_resolution(app, desc->width, desc->height);

    app->is_initialized = true;
    return 0;
}

void mf_host_app_set_time(mf_host_app* app, float current_time) {
    if (!app || !app->is_initialized) return;
    if (!app->resources.time) app->resources.time = mf_engine_map_resource(app->engine, "u_Time");
    
    if (app->resources.time) {
        f32* d = (f32*)mf_tensor_data(app->resources.time);
        if (d) *d = current_time;
    }
}

void mf_host_app_set_mouse(mf_host_app* app, float x, float y, bool lmb, bool rmb) {
    if (!app || !app->is_initialized) return;
    if (!app->resources.mouse) app->resources.mouse = mf_engine_map_resource(app->engine, "u_Mouse");

    if (app->resources.mouse) {
        f32* d = (f32*)mf_tensor_data(app->resources.mouse);
        if (d) {
            d[0] = x;
            d[1] = y;
            d[2] = lmb ? 1.0f : 0.0f;
            d[3] = rmb ? 1.0f : 0.0f;
        }
    }

    // Individual mouse coords if available (optional)
    mf_tensor* t_mx = mf_engine_map_resource(app->engine, "u_MouseX");
    if (t_mx) { f32* d = mf_tensor_data(t_mx); if (d) *d = x; }
    mf_tensor* t_my = mf_engine_map_resource(app->engine, "u_MouseY");
    if (t_my) { f32* d = mf_tensor_data(t_my); if (d) *d = y; }
}

void mf_host_app_set_resolution(mf_host_app* app, int width, int height) {
    if (!app || !app->engine) return;

    app->desc.width = width;
    app->desc.height = height;

    int32_t screen_shape[] = { height, width, 4 };
    mf_engine_resize_resource(app->engine, "out_Color", screen_shape, 3);

    mf_tensor* t_res = mf_engine_map_resource(app->engine, "u_Resolution");
    if (t_res) {
        f32* d = (f32*)mf_tensor_data(t_res);
        if (d && mf_tensor_count(t_res) >= 2) {
            d[0] = (f32)width;
            d[1] = (f32)height;
        }
    }

    // Update individual resolution uniforms if they exist
    mf_tensor* t_rx = mf_engine_map_resource(app->engine, "u_ResX");
    if (t_rx) { f32* d = mf_tensor_data(t_rx); if (d) *d = (f32)width; }
    mf_tensor* t_ry = mf_engine_map_resource(app->engine, "u_ResY");
    if (t_ry) { f32* d = mf_tensor_data(t_ry); if (d) *d = (f32)height; }
    mf_tensor* t_aspect = mf_engine_map_resource(app->engine, "u_Aspect");
    if (t_aspect) { f32* d = mf_tensor_data(t_aspect); if (d) *d = (f32)width / (f32)height; }
}

void mf_host_app_cleanup(mf_host_app* app) {
    if (!app) return;
    if (app->engine) {
        mf_engine_destroy(app->engine);
    }
    // We don't cleanup the descriptor here because mf_host_app_init
    // only did a shallow copy, and the caller is responsible for the original descriptor.
    memset(app, 0, sizeof(mf_host_app));
}
