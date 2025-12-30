#include <mathflow/host/mf_manifest_loader.h>
#include <mathflow/base/mf_log.h>
#include <cjson/cJSON.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define PATH_SEP '\\'
#else
#define PATH_SEP '/'
#endif

// --- File Utils ---

static char* read_file_content(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* buf = (char*)malloc(len + 1);
    if (buf) {
        if (fread(buf, 1, len, f) != (size_t)len) {
            free(buf);
            buf = NULL;
        } else {
            buf[len] = '\0';
        }
    }
    fclose(f);
    return buf;
}

static char* get_dir(const char* path) {
    const char* last_slash = strrchr(path, '/');
#ifdef _WIN32
    const char* last_bslash = strrchr(path, '\\');
    if (last_bslash > last_slash) last_slash = last_bslash;
#endif

    if (!last_slash) return strdup(".");

    size_t len = last_slash - path;
    char* dir = (char*)malloc(len + 1);
    memcpy(dir, path, len);
    dir[len] = '\0';
    return dir;
}

static char* join_path(const char* dir, const char* file) {
    if (file[0] == '/' || file[0] == '\\' || (strlen(file) > 2 && file[1] == ':')) {
        return strdup(file);
    }

    size_t len1 = strlen(dir);
    size_t len2 = strlen(file);
    char* path = (char*)malloc(len1 + len2 + 2);
    
    bool slash = (len1 > 0 && (dir[len1-1] == '/' || dir[len1-1] == '\\'));
    
    if (slash) sprintf(path, "%s%s", dir, file);
    else sprintf(path, "%s/%s", dir, file);
    
    return path;
}

static mf_dtype parse_dtype(const char* s) {
    if (!s) return MF_DTYPE_F32;
    if (strcmp(s, "F32") == 0) return MF_DTYPE_F32;
    if (strcmp(s, "I32") == 0) return MF_DTYPE_I32;
    if (strcmp(s, "U8") == 0) return MF_DTYPE_U8;
    return MF_DTYPE_F32;
}

// --- Loader ---

int mf_app_load_config(const char* mfapp_path, mf_host_desc* out_desc) {
    char* json_str = read_file_content(mfapp_path);
    if (!json_str) {
        MF_LOG_ERROR("Could not read manifest %s", mfapp_path);
        return -1;
    }

    cJSON* root = cJSON_Parse(json_str);
    if (!root) {
        MF_LOG_ERROR("Failed to parse manifest JSON");
        free(json_str);
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

    char* base_dir = get_dir(mfapp_path);

    // 1. Runtime Section
    cJSON* runtime = cJSON_GetObjectItem(root, "runtime");
    if (runtime) {
        cJSON* entry = cJSON_GetObjectItem(runtime, "entry");
        if (entry && cJSON_IsString(entry)) {
            out_desc->graph_path = join_path(base_dir, entry->valuestring);
        }
        
        cJSON* threads = cJSON_GetObjectItem(runtime, "threads");
        if (threads && cJSON_IsNumber(threads)) out_desc->num_threads = threads->valueint;
    }

    // 2. Window Section
    cJSON* window = cJSON_GetObjectItem(root, "window");
    if (window) {
        cJSON* title = cJSON_GetObjectItem(window, "title");
        if (title && cJSON_IsString(title)) {
            out_desc->window_title = strdup(title->valuestring);
        }

        cJSON* w = cJSON_GetObjectItem(window, "width");
        if (w && cJSON_IsNumber(w)) out_desc->width = w->valueint;

        cJSON* h = cJSON_GetObjectItem(window, "height");
        if (h && cJSON_IsNumber(h)) out_desc->height = h->valueint;

        cJSON* resizable = cJSON_GetObjectItem(window, "resizable");
        if (resizable && cJSON_IsBool(resizable)) out_desc->resizable = cJSON_IsTrue(resizable);

        cJSON* vsync = cJSON_GetObjectItem(window, "vsync");
        if (vsync && cJSON_IsBool(vsync)) out_desc->vsync = cJSON_IsTrue(vsync);

        cJSON* fullscreen = cJSON_GetObjectItem(window, "fullscreen");
        if (fullscreen && cJSON_IsBool(fullscreen)) out_desc->fullscreen = cJSON_IsTrue(fullscreen);
    }

    // 3. Pipeline Section
    cJSON* pipeline = cJSON_GetObjectItem(root, "pipeline");
    if (pipeline) {
        out_desc->has_pipeline = true;
        
        // Resources
        cJSON* resources = cJSON_GetObjectItem(pipeline, "resources");
        if (cJSON_IsArray(resources)) {
            out_desc->pipeline.resource_count = cJSON_GetArraySize(resources);
            out_desc->pipeline.resources = calloc(out_desc->pipeline.resource_count, sizeof(mf_pipeline_resource));
            
            int i = 0;
            cJSON* res = NULL;
            cJSON_ArrayForEach(res, resources) {
                mf_pipeline_resource* pr = &out_desc->pipeline.resources[i++];
                cJSON* name = cJSON_GetObjectItem(res, "name");
                if (name) pr->name = strdup(name->valuestring);
                
                pr->dtype = parse_dtype(cJSON_GetStringValue(cJSON_GetObjectItem(res, "dtype")));
                
                cJSON* shape = cJSON_GetObjectItem(res, "shape");
                if (cJSON_IsArray(shape)) {
                    pr->ndim = (uint8_t)cJSON_GetArraySize(shape);
                    for(int d=0; d < pr->ndim && d < MF_MAX_DIMS; ++d) {
                        pr->shape[d] = cJSON_GetArrayItem(shape, d)->valueint;
                    }
                }

                cJSON* pers = cJSON_GetObjectItem(res, "persistent");
                if (cJSON_IsBool(pers)) pr->persistent = cJSON_IsTrue(pers);
            }
        }

        // Kernels
        cJSON* kernels = cJSON_GetObjectItem(pipeline, "kernels");
        if (cJSON_IsArray(kernels)) {
            out_desc->pipeline.kernel_count = cJSON_GetArraySize(kernels);
            out_desc->pipeline.kernels = calloc(out_desc->pipeline.kernel_count, sizeof(mf_pipeline_kernel));

            int i = 0;
            cJSON* ker = NULL;
            cJSON_ArrayForEach(ker, kernels) {
                mf_pipeline_kernel* pk = &out_desc->pipeline.kernels[i++];
                
                cJSON* id = cJSON_GetObjectItem(ker, "id");
                if (id && cJSON_IsString(id)) pk->id = strdup(id->valuestring);

                cJSON* entry = cJSON_GetObjectItem(ker, "entry");
                if (entry && cJSON_IsString(entry)) pk->graph_path = join_path(base_dir, entry->valuestring);

                cJSON* freq = cJSON_GetObjectItem(ker, "frequency");
                if (freq && cJSON_IsNumber(freq)) pk->frequency = (u32)freq->valueint;
                else pk->frequency = 1;

                cJSON* bindings = cJSON_GetObjectItem(ker, "bindings");
                if (bindings && cJSON_IsArray(bindings)) {
                    pk->binding_count = cJSON_GetArraySize(bindings);
                    pk->bindings = calloc(pk->binding_count, sizeof(mf_pipeline_binding));
                    
                    int b = 0;
                    cJSON* bind = NULL;
                    cJSON_ArrayForEach(bind, bindings) {
                        mf_pipeline_binding* pb = &pk->bindings[b++];
                        // Expected format: "port_name": "resource_name" (Key-Value)
                        // But cJSON array of objects? Or object?
                        // "bindings": [ {"port": "A", "resource": "B"}, ... ]
                        // Let's assume array of objects based on previous context.
                        cJSON* port = cJSON_GetObjectItem(bind, "port");
                        cJSON* res = cJSON_GetObjectItem(bind, "resource");
                        if (port) pb->kernel_port = strdup(port->valuestring);
                        if (res) pb->global_resource = strdup(res->valuestring);
                    }
                }
            }
        }
    }

    free(base_dir);
    cJSON_Delete(root);
    free(json_str);
    
    if (!out_desc->graph_path && !out_desc->has_pipeline) {
        MF_LOG_ERROR("Manifest missing runtime.entry or pipeline section");
        return -3;
    }

    return 0;
}
