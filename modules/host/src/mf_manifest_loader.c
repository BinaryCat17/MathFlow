#include <mathflow/host/mf_manifest_loader.h>
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
    // If file is absolute, return duplicate
    if (file[0] == '/' || file[0] == '\\' || (strlen(file) > 2 && file[1] == ':')) {
        return strdup(file);
    }

    size_t len1 = strlen(dir);
    size_t len2 = strlen(file);
    char* path = (char*)malloc(len1 + len2 + 2);
    
    bool slash = (len1 > 0 && (dir[len1-1] == '/' || dir[len1-1] == '\\'));
    
    if (slash) sprintf(path, "%s%s", dir, file);
    else sprintf(path, "%s/%s", dir, file);
    
    // Normalize slashes to platform? 
    // For now, mixed slashes usually work on Windows C APIs, but let's be safe later if needed.
    // The previous implementation didn't normalize, so we stick to that.
    
    return path;
}

// --- Loader ---

int mf_app_load_config(const char* mfapp_path, mf_host_desc* out_desc) {
    char* json_str = read_file_content(mfapp_path);
    if (!json_str) {
        fprintf(stderr, "Error: Could not read manifest %s\n", mfapp_path);
        return -1;
    }

    cJSON* root = cJSON_Parse(json_str);
    if (!root) {
        fprintf(stderr, "Error: Failed to parse manifest JSON\n");
        free(json_str);
        return -2;
    }

    // Defaults
    out_desc->num_threads = 0;
    out_desc->fullscreen = false;
    out_desc->resizable = true;
    out_desc->vsync = true;
    out_desc->runtime_type = MF_HOST_RUNTIME_SHADER;
    out_desc->width = 800;
    out_desc->height = 600;
    out_desc->window_title = "MathFlow App";

    // 1. Runtime Section
    cJSON* runtime = cJSON_GetObjectItem(root, "runtime");
    if (runtime) {
        cJSON* type = cJSON_GetObjectItem(runtime, "type");
        if (type && cJSON_IsString(type)) {
            if (strcmp(type->valuestring, "script") == 0) {
                out_desc->runtime_type = MF_HOST_RUNTIME_SCRIPT;
            } else {
                out_desc->runtime_type = MF_HOST_RUNTIME_SHADER;
            }
        }

        cJSON* entry = cJSON_GetObjectItem(runtime, "entry");
        if (entry && cJSON_IsString(entry)) {
            // Resolve path relative to manifest
            char* base_dir = get_dir(mfapp_path);
            out_desc->graph_path = join_path(base_dir, entry->valuestring);
            free(base_dir);
        }
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

    cJSON_Delete(root);
    free(json_str);
    
    if (!out_desc->graph_path) {
        fprintf(stderr, "Error: Manifest missing runtime.entry\n");
        return -3;
    }

    return 0;
}
