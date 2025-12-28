#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/host/mf_manifest_loader.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: mf-window <app.mfapp>\n");
        return 1;
    }

    const char* mfapp_path = argv[1];
    mf_host_desc desc = {0};

    if (mf_app_load_config(mfapp_path, &desc) != 0) {
        const char* ext = strrchr(mfapp_path, '.');
        if (ext && (strcmp(ext, ".json") == 0 || strcmp(ext, ".bin") == 0)) {
            printf("Warning: Loading raw graph directly.\n");
            desc.graph_path = mfapp_path;
            desc.window_title = "MathFlow Visualizer";
            desc.width = 800;
            desc.height = 600;
            desc.resizable = true;
            desc.vsync = true;
        } else {
            printf("Error: Failed to load manifest %s\n", mfapp_path);
            return 1;
        }
    }

    printf("MathFlow Visualizer\n");
    printf("App: %s\n", desc.window_title);
    printf("Graph: %s\n", desc.graph_path);
    printf("Resolution: %dx%d\n", desc.width, desc.height);

    return mf_host_run(&desc);
}