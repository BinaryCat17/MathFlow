#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/host/mf_manifest_loader.h>
#include <mathflow/host/mf_asset_loader.h>
#include <mathflow/host/mf_host_headless.h>

static void print_help(const char* prog) {
    printf("Usage: %s <app.mfapp> [options]\n", prog);
    printf("Options:\n");
    printf("  --frames <n>   Number of frames to execute (default: 1)\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    const char* mfapp_path = argv[1];
    int frames = 1;
    
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frames = atoi(argv[i+1]);
            i++;
        }
    }

    // 1. Load Manifest
    mf_host_desc app_desc = {0};
    if (mf_app_load_config(mfapp_path, &app_desc) != 0) {
        // Fallback: Try loading as raw graph if not mfapp
        const char* ext = strrchr(mfapp_path, '.');
        if (ext && (strcmp(ext, ".json") == 0 || strcmp(ext, ".bin") == 0)) {
            printf("Warning: Loading raw graph directly. Consider using .mfapp.\n");
            app_desc.graph_path = mfapp_path;
            app_desc.window_title = "Raw Graph";
        } else {
             printf("Error: Failed to load manifest %s\n", mfapp_path);
             return 1;
        }
    }

    printf("MathFlow Runner\n");
    printf("App: %s\n", app_desc.window_title);
    printf("Graph: %s\n", app_desc.graph_path);
    printf("Mode: %s\n", app_desc.runtime_type == MF_HOST_RUNTIME_SHADER ? "Shader (Simulated)" : "Script");

    // 2. Initialize Engine
    mf_engine engine;
    mf_engine_init(&engine, NULL);

    // 3. Load Graph using Asset Loader
    if (!mf_asset_loader_load(&engine, app_desc.graph_path)) {
        printf("Error: Failed to load graph.\n");
        mf_engine_shutdown(&engine);
        return 1;
    }

    printf("Program: %u tensors, %u insts\n", engine.program->meta.tensor_count, engine.program->meta.instruction_count);

    // 4. Run Headless
    int result = mf_host_run_headless(&engine, frames);

    // 5. Cleanup
    mf_engine_shutdown(&engine);
    
    return result;
}