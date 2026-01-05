#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_memory.h>
#include <mathflow/base/mf_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage() {
    printf("MathFlow Cartridge Compiler (mfc) v1.3\n");
    printf("Usage: mfc <input.mfapp|input.json> [output.mfc]\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char* input_path = argv[1];
    char output_path[256];
    if (argc >= 3) {
        strncpy(output_path, argv[2], 255);
    } else {
        strncpy(output_path, input_path, 250);
        char* ext = strrchr(output_path, '.');
        if (ext) *ext = '\0';
        strcat(output_path, ".mfc");
    }

    size_t arena_size = 1024 * 1024 * 128; // 128MB
    void* backing = malloc(arena_size);
    mf_arena arena;
    mf_arena_init(&arena, backing, arena_size);

    mf_section_desc sections[MF_MAX_SECTIONS];
    u32 section_count = 0;
    mf_graph_ir app_ir = {0};

    const char* ext = mf_path_get_ext(input_path);
    bool success = false;

    if (strcmp(ext, "mfapp") == 0) {
        mf_compiler_manifest manifest;
        if (mf_compiler_load_manifest(input_path, &manifest, &arena)) {
            app_ir = manifest.app_ir;
            for (u32 i = 0; i < manifest.kernel_count; ++i) {
                MF_LOG_INFO("Compiling kernel \'%s\'...", manifest.kernels[i].id);
                mf_compiler_diag diag; mf_compiler_diag_init(&diag, &arena);
                mf_graph_ir k_ir = {0};
                if (mf_compile_load_json(manifest.kernels[i].path, &k_ir, &arena, &diag)) {
                    mf_program* prog = mf_compile(&k_ir, &arena, &diag);
                    if (prog) {
                        sections[section_count++] = (mf_section_desc){ manifest.kernels[i].id, MF_SECTION_PROGRAM, prog, 0 };
                    } else success = false;
                } else success = false;
            }
            // Embed assets
            for (u32 i = 0; i < manifest.asset_count; ++i) {
                size_t f_size = 0;
                void* f_data = mf_file_read_bin(manifest.assets[i].path, &f_size);
                if (f_data) {
                    void* arena_data = mf_arena_alloc((mf_allocator*)&arena, f_size);
                    memcpy(arena_data, f_data, f_size);
                    free(f_data);
                    sections[section_count++] = (mf_section_desc){ manifest.assets[i].name, manifest.assets[i].type, arena_data, (u32)f_size };
                    MF_LOG_INFO("Embedded asset \'%s\'", manifest.assets[i].name);
                }
            }
            // Embed pipeline
            sections[section_count++] = (mf_section_desc){ "pipeline", MF_SECTION_PIPELINE, manifest.raw_json, manifest.raw_json_size };
            success = true;
        }
    } else {
        MF_LOG_INFO("Compiling single graph %s...", input_path);
        mf_compiler_diag diag;
        mf_compiler_diag_init(&diag, &arena);
        if (mf_compile_load_json(input_path, &app_ir, &arena, &diag)) {
            mf_program* prog = mf_compile(&app_ir, &arena, &diag);
            if (prog) {
                sections[section_count++] = (mf_section_desc){ "main", MF_SECTION_PROGRAM, prog, 0 };
                success = true;
            }
        }
    }

    if (success) {
        if (!mf_compile_save_cartridge(output_path, &app_ir, sections, section_count)) {
            MF_LOG_ERROR("Failed to save cartridge.");
            success = false;
        } else {
            MF_LOG_INFO("Successfully created cartridge: %s", output_path);
        }
    }

    free(backing);
    return success ? 0 : 1;
}
