#include <mathflow/compiler/mf_compiler.h>
#include "mf_compiler_internal.h"
#include <mathflow/base/mf_json.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <string.h>
#include <stdlib.h>

bool mf_compiler_load_manifest(const char* path, mf_compiler_manifest* out_manifest, mf_arena* arena) {
    if (!path || !out_manifest) return false;
    memset(out_manifest, 0, sizeof(mf_compiler_manifest));

    char* json_str = mf_file_read(path, arena);
    if (!json_str) return false;

    mf_json_value* root = mf_json_parse(json_str, arena);
    if (!root || root->type != MF_JSON_VAL_OBJECT) return false;

    mf_ir_parse_window_settings(root, &out_manifest->app_ir);
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

    // Case C: Raw Graph (no pipeline/runtime, but has nodes)
    if (out_manifest->kernel_count == 0) {
        const mf_json_value* nodes = mf_json_get_field(root, "nodes");
        if (nodes && nodes->type == MF_JSON_VAL_ARRAY) {
            out_manifest->kernel_count = 1;
            out_manifest->kernels = MF_ARENA_PUSH(arena, mf_compiler_kernel_desc, 1);
            out_manifest->kernels[0].id = "main";
            out_manifest->kernels[0].path = mf_arena_strdup(arena, path);
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
