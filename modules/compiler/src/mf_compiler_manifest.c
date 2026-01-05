#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/base/mf_json.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <string.h>
#include <stdlib.h>

// --- Internal Manifest Parser ---

static void _parse_window_settings(const mf_json_value* root, mf_graph_ir* out_ir) {
    const mf_json_value* window = mf_json_get_field(root, "window");
    if (!window || window->type != MF_JSON_VAL_OBJECT) return;

    const mf_json_value* title = mf_json_get_field(window, "title");
    if (title && title->type == MF_JSON_VAL_STRING) {
        strncpy(out_ir->app_title, title->as.s, MF_MAX_TITLE_NAME - 1);
    }

    const mf_json_value* w = mf_json_get_field(window, "width");
    if (w && w->type == MF_JSON_VAL_NUMBER) out_ir->window_width = (u32)w->as.n;

    const mf_json_value* h = mf_json_get_field(window, "height");
    if (h && h->type == MF_JSON_VAL_NUMBER) out_ir->window_height = (u32)h->as.n;

    const mf_json_value* resizable = mf_json_get_field(window, "resizable");
    if (resizable && resizable->type == MF_JSON_VAL_BOOL) out_ir->resizable = resizable->as.b;

    const mf_json_value* vsync = mf_json_get_field(window, "vsync");
    if (vsync && vsync->type == MF_JSON_VAL_BOOL) out_ir->vsync = vsync->as.b;

    const mf_json_value* fullscreen = mf_json_get_field(window, "fullscreen");
    if (fullscreen && fullscreen->type == MF_JSON_VAL_BOOL) out_ir->fullscreen = fullscreen->as.b;
}

bool mf_compiler_load_manifest(const char* path, mf_compiler_manifest* out_manifest, mf_arena* arena) {
    if (!path || !out_manifest) return false;
    memset(out_manifest, 0, sizeof(mf_compiler_manifest));

    char* json_str = mf_file_read(path, arena);
    if (!json_str) return false;

    mf_json_value* root = mf_json_parse(json_str, arena);
    if (!root || root->type != MF_JSON_VAL_OBJECT) return false;

    _parse_window_settings(root, &out_manifest->app_ir);
    char* base_dir = mf_path_get_dir(path, arena);

    // 1. Kernels
    const mf_json_value* pipeline = mf_json_get_field(root, "pipeline");
    const mf_json_value* runtime = mf_json_get_field(root, "runtime");

    // Case A: Simple single-kernel app (runtime.entry)
    if (runtime && runtime->type == MF_JSON_VAL_OBJECT && !pipeline) {
        const mf_json_value* entry = mf_json_get_field(runtime, "entry");
        if (entry && entry->type == MF_JSON_VAL_STRING) {
            out_manifest->kernel_count = 1;
            out_manifest->kernels = MF_ARENA_PUSH(arena, mf_compiler_kernel_desc, 1);
            out_manifest->kernels[0].id = "main";
            out_manifest->kernels[0].path = mf_path_join(base_dir, entry->as.s, arena);
        }
    }

    // Case B: Multi-kernel pipeline
    if (pipeline && pipeline->type == MF_JSON_VAL_OBJECT) {
        const mf_json_value* kernels = mf_json_get_field(pipeline, "kernels");
        if (kernels && kernels->type == MF_JSON_VAL_ARRAY) {
            out_manifest->kernel_count = (u32)kernels->as.array.count;
            out_manifest->kernels = MF_ARENA_PUSH(arena, mf_compiler_kernel_desc, out_manifest->kernel_count);
            for (size_t i = 0; i < kernels->as.array.count; ++i) {
                const mf_json_value* k = &kernels->as.array.items[i];
                const mf_json_value* id = mf_json_get_field(k, "id");
                const mf_json_value* entry = mf_json_get_field(k, "entry");
                out_manifest->kernels[i].id = id ? id->as.s : "kernel";
                out_manifest->kernels[i].path = entry ? mf_path_join(base_dir, entry->as.s, arena) : NULL;
            }
        }
    }

    // 2. Assets
    const mf_json_value* assets = mf_json_get_field(root, "assets");
    if (assets && assets->type == MF_JSON_VAL_ARRAY) {
        out_manifest->asset_count = (u32)assets->as.array.count;
        out_manifest->assets = MF_ARENA_PUSH(arena, mf_compiler_asset_desc, out_manifest->asset_count);
        for (size_t i = 0; i < assets->as.array.count; ++i) {
            const mf_json_value* a = &assets->as.array.items[i];
            const mf_json_value* name = mf_json_get_field(a, "name");
            const mf_json_value* a_path = mf_json_get_field(a, "path");
            const mf_json_value* type = mf_json_get_field(a, "type");
            
            out_manifest->assets[i].name = name ? name->as.s : "asset";
            out_manifest->assets[i].path = a_path ? mf_path_join(base_dir, a_path->as.s, arena) : NULL;
            
            if (type && type->type == MF_JSON_VAL_STRING) {
                if (strcmp(type->as.s, "image") == 0) out_manifest->assets[i].type = MF_SECTION_IMAGE;
                else if (strcmp(type->as.s, "font") == 0) out_manifest->assets[i].type = MF_SECTION_FONT;
                else out_manifest->assets[i].type = MF_SECTION_RAW;
            } else {
                out_manifest->assets[i].type = MF_SECTION_RAW;
            }
        }
    }

    // Store the raw JSON for the PIPELINE section
    out_manifest->raw_json = json_str;
    out_manifest->raw_json_size = (u32)strlen(json_str);

    return true;
}
