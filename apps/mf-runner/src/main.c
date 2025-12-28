#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mathflow/host/mf_manifest_loader.h>
#include <mathflow/host/mf_host_headless.h>
#include <mathflow/host/mf_host_desc.h>

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

    mf_host_desc app_desc = {0};
    if (mf_app_load_config(mfapp_path, &app_desc) != 0) {
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

    int result = mf_host_run_headless(&app_desc, frames);

    return result;
}