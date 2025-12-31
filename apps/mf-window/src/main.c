#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/host/mf_manifest_loader.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_platform.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv) {
    // Clear old logs first
    if (mf_fs_mkdir("logs")) {
        mf_fs_clear_dir("logs");
    }

    mf_log_init();

    // Setup File Logging
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char log_path[256];
    // Create logs directory just in case (system specific, but assuming pre-existence or simple fail)
    // Using a timestamp format for unique logs
    strftime(log_path, sizeof(log_path), "logs/log_%Y-%m-%d_%H-%M-%S.txt", t);
    
    // Log everything (TRACE) to file, while Console keeps its default (INFO) or user config
    mf_log_add_file_sink(log_path, MF_LOG_LEVEL_TRACE);

    if (argc < 2) {
        printf("Usage: mf-window <app.mfapp>\n");
        return 1;
    }

    const char* mfapp_path = argv[1];
    mf_host_desc desc = {0};
    desc.log_interval = 5.0f; // Default 5 seconds

    // Simple Argument Parser
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--log-interval") == 0 && i + 1 < argc) {
            desc.log_interval = (float)atof(argv[++i]);
        }
    }

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