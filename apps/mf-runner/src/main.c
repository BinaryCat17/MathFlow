#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mathflow/host/mf_manifest_loader.h>
#include <mathflow/host/mf_host_headless.h>
#include <mathflow/host/mf_host_desc.h>
#include <mathflow/base/mf_log.h>

static void print_help(const char* prog) {
    printf("Usage: %s <app.mfapp> [options]\n", prog);
    printf("Options:\n");
    printf("  --frames <n>   Number of frames to execute (default: 1)\n");
    printf("  --trace        Enable trace logging\n");
}

int main(int argc, char** argv) {
    // Init Logging
    mf_log_init();

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
        else if (strcmp(argv[i], "--trace") == 0) {
            mf_log_set_global_level(MF_LOG_LEVEL_TRACE);
        }
    }

    mf_host_desc app_desc = {0};
    if (mf_app_load_config(mfapp_path, &app_desc) != 0) {
        const char* ext = strrchr(mfapp_path, '.');
        if (ext && (strcmp(ext, ".json") == 0 || strcmp(ext, ".bin") == 0)) {
            MF_LOG_WARN("Loading raw graph directly. Consider using .mfapp.");
            app_desc.has_pipeline = true;
            app_desc.pipeline.kernel_count = 1;
            app_desc.pipeline.kernels = calloc(1, sizeof(mf_pipeline_kernel));
            app_desc.pipeline.kernels[0].id = strdup("main");
            app_desc.pipeline.kernels[0].graph_path = strdup(mfapp_path);
            app_desc.pipeline.kernels[0].frequency = 1;
            app_desc.window_title = "Raw Graph";
        } else {
             MF_LOG_ERROR("Failed to load manifest %s", mfapp_path);
             return 1;
        }
    }

    const char* entry_info = app_desc.has_pipeline ? app_desc.pipeline.kernels[0].graph_path : "Pipeline";
    MF_LOG_INFO("MathFlow Runner | App: %s | Entry: %s", app_desc.window_title, entry_info);

    int result = mf_host_run_headless(&app_desc, frames);

    return result;
}