#include <mathflow/host/mf_manifest_loader.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_json.h>
#include <mathflow/base/mf_memory.h>
#include <mathflow/base/mf_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mf_dtype parse_dtype_str(const char* s) {
    if (!s) return MF_DTYPE_F32;
    if (strcmp(s, "F32") == 0) return MF_DTYPE_F32;
    if (strcmp(s, "I32") == 0) return MF_DTYPE_I32;
    if (strcmp(s, "U8") == 0) return MF_DTYPE_U8;
    return MF_DTYPE_F32;
}

// --- Loader ---

int mf_app_load_config(const char* mfapp_path, mf_host_desc* out_desc) {
    // Temp Arena for Parsing and file reading
    size_t arena_size = 1024 * 1024; // 1MB should be enough for manifest
    void* arena_mem = malloc(arena_size);
    mf_arena arena;
    mf_arena_init(&arena, arena_mem, arena_size);

    char* json_str = mf_file_read(mfapp_path, &arena);
    if (!json_str) {
        MF_LOG_ERROR("Could not read manifest %s", mfapp_path);
        free(arena_mem);
        return -1;
    }

    mf_json_value* root = mf_json_parse(json_str, &arena);
    if (!root || root->type != MF_JSON_VAL_OBJECT) {
        MF_LOG_ERROR("Failed to parse manifest JSON");
        free(arena_mem);
        return -2;
    }

    // Defaults
    out_desc->num_threads = 0;
    out_desc->fullscreen = false;
    out_desc->resizable = true;
    out_desc->vsync = true;
    out_desc->width = 800;
    out_desc->height = 600;
    out_desc->window_title = "MathFlow App";
    out_desc->graph_path = NULL;
    out_desc->has_pipeline = false;
    memset(&out_desc->pipeline, 0, sizeof(out_desc->pipeline));

    char* base_dir = mf_path_get_dir(mfapp_path, &arena);

    // 1. Runtime Section
    const mf_json_value* runtime = mf_json_get_field(root, "runtime");
    if (runtime && runtime->type == MF_JSON_VAL_OBJECT) {
        const mf_json_value* entry = mf_json_get_field(runtime, "entry");
        if (entry && entry->type == MF_JSON_VAL_STRING) {
            char* path = mf_path_join(base_dir, entry->as.s, &arena);
            out_desc->graph_path = strdup(path);
        }
        
        const mf_json_value* threads = mf_json_get_field(runtime, "threads");
        if (threads && threads->type == MF_JSON_VAL_NUMBER) out_desc->num_threads = (u32)threads->as.n;
    }

    // 2. Window Section
    const mf_json_value* window = mf_json_get_field(root, "window");
    if (window && window->type == MF_JSON_VAL_OBJECT) {
        const mf_json_value* title = mf_json_get_field(window, "title");
        if (title && title->type == MF_JSON_VAL_STRING) {
            out_desc->window_title = strdup(title->as.s);
        }

        const mf_json_value* w = mf_json_get_field(window, "width");
        if (w && w->type == MF_JSON_VAL_NUMBER) out_desc->width = (u32)w->as.n;

        const mf_json_value* h = mf_json_get_field(window, "height");
        if (h && h->type == MF_JSON_VAL_NUMBER) out_desc->height = (u32)h->as.n;

        const mf_json_value* resizable = mf_json_get_field(window, "resizable");
        if (resizable && resizable->type == MF_JSON_VAL_BOOL) out_desc->resizable = resizable->as.b;

        const mf_json_value* vsync = mf_json_get_field(window, "vsync");
        if (vsync && vsync->type == MF_JSON_VAL_BOOL) out_desc->vsync = vsync->as.b;

        const mf_json_value* fullscreen = mf_json_get_field(window, "fullscreen");
        if (fullscreen && fullscreen->type == MF_JSON_VAL_BOOL) out_desc->fullscreen = fullscreen->as.b;
    }

    // 3. Pipeline Section
    const mf_json_value* pipeline = mf_json_get_field(root, "pipeline");
    if (pipeline && pipeline->type == MF_JSON_VAL_OBJECT) {
        out_desc->has_pipeline = true;
        
        // Resources
        const mf_json_value* resources = mf_json_get_field(pipeline, "resources");
        if (resources && resources->type == MF_JSON_VAL_ARRAY) {
            out_desc->pipeline.resource_count = (u32)resources->as.array.count;
            out_desc->pipeline.resources = calloc(out_desc->pipeline.resource_count, sizeof(mf_pipeline_resource));
            
            for (size_t i = 0; i < resources->as.array.count; ++i) {
                mf_pipeline_resource* pr = &out_desc->pipeline.resources[i];
                const mf_json_value* res = &resources->as.array.items[i];
                
                const mf_json_value* name = mf_json_get_field(res, "name");
                if (name && name->type == MF_JSON_VAL_STRING) pr->name = strdup(name->as.s);
                
                const mf_json_value* dtype = mf_json_get_field(res, "dtype");
                if (dtype && dtype->type == MF_JSON_VAL_STRING) pr->dtype = parse_dtype_str(dtype->as.s);
                
                const mf_json_value* shape = mf_json_get_field(res, "shape");
                if (shape && shape->type == MF_JSON_VAL_ARRAY) {
                    pr->ndim = (uint8_t)shape->as.array.count;
                    if (pr->ndim > MF_MAX_DIMS) pr->ndim = MF_MAX_DIMS;
                    for(int d=0; d < pr->ndim; ++d) {
                        const mf_json_value* dim = &shape->as.array.items[d];
                        if (dim->type == MF_JSON_VAL_NUMBER) pr->shape[d] = (int)dim->as.n;
                    }
                }
            }
        }

        // Kernels
        const mf_json_value* kernels = mf_json_get_field(pipeline, "kernels");
        if (kernels && kernels->type == MF_JSON_VAL_ARRAY) {
            out_desc->pipeline.kernel_count = (u32)kernels->as.array.count;
            out_desc->pipeline.kernels = calloc(out_desc->pipeline.kernel_count, sizeof(mf_pipeline_kernel));

            for (size_t i = 0; i < kernels->as.array.count; ++i) {
                mf_pipeline_kernel* pk = &out_desc->pipeline.kernels[i];
                const mf_json_value* ker = &kernels->as.array.items[i];
                
                const mf_json_value* id = mf_json_get_field(ker, "id");
                if (id && id->type == MF_JSON_VAL_STRING) pk->id = strdup(id->as.s);

                const mf_json_value* entry = mf_json_get_field(ker, "entry");
                if (entry && entry->type == MF_JSON_VAL_STRING) {
                    char* path = mf_path_join(base_dir, entry->as.s, &arena);
                    pk->graph_path = strdup(path);
                }

                const mf_json_value* freq = mf_json_get_field(ker, "frequency");
                if (freq && freq->type == MF_JSON_VAL_NUMBER) pk->frequency = (u32)freq->as.n;
                else pk->frequency = 1;

                const mf_json_value* bindings = mf_json_get_field(ker, "bindings");
                if (bindings && bindings->type == MF_JSON_VAL_ARRAY) {
                    pk->binding_count = (u32)bindings->as.array.count;
                    pk->bindings = calloc(pk->binding_count, sizeof(mf_pipeline_binding));
                    
                    for (size_t b = 0; b < bindings->as.array.count; ++b) {
                        mf_pipeline_binding* pb = &pk->bindings[b];
                        const mf_json_value* bind = &bindings->as.array.items[b];
                        
                        const mf_json_value* port = mf_json_get_field(bind, "port");
                        const mf_json_value* res = mf_json_get_field(bind, "resource");
                        if (port && port->type == MF_JSON_VAL_STRING) pb->kernel_port = strdup(port->as.s);
                        if (res && res->type == MF_JSON_VAL_STRING) pb->global_resource = strdup(res->as.s);
                    }
                }
            }
        }
    }

    // Cleanup
    free(arena_mem);
    
    if (!out_desc->graph_path && !out_desc->has_pipeline) {
        MF_LOG_ERROR("Manifest missing runtime.entry or pipeline section");
        return -3;
    }

    return 0;
}
