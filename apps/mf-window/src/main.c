#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/host/mf_manifest_loader.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    mf_log_init();

    if (argc < 2) {
        printf("Usage: mf-window <app.mfapp>\n");
        return 1;
    }

    const char* mfapp_path = argv[1];
    mf_host_desc desc = {0};

    if (mf_app_load_config(mfapp_path, &desc) != 0) {
        const char* ext = strrchr(mfapp_path, '.');
        if (ext && (strcmp(ext, ".json") == 0 || strcmp(ext, ".bin") == 0)) {
            MF_LOG_WARN("Loading raw graph directly.");
            desc.graph_path = mfapp_path;
            desc.window_title = "MathFlow Visualizer";
            desc.width = 800;
            desc.height = 600;
            desc.resizable = true;
            desc.vsync = true;
        } else {
            MF_LOG_ERROR("Failed to load manifest %s", mfapp_path);
            return 1;
        }
    }

    MF_LOG_INFO("MathFlow Visualizer");
    MF_LOG_INFO("App: %s", desc.window_title);
    MF_LOG_INFO("Graph: %s", desc.graph_path);
    MF_LOG_INFO("Resolution: %dx%d", desc.width, desc.height);

    return mf_host_run(&desc);
}