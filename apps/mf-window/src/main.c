#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/host/mf_host_desc.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_platform.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv) {
    if (mf_fs_mkdir("logs")) {
        mf_fs_clear_dir("logs");
    }

    mf_log_init();

    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char log_path[256];
    strftime(log_path, sizeof(log_path), "logs/log_%Y-%m-%d_%H-%M-%S.txt", t);
    mf_log_add_file_sink(log_path, MF_LOG_LEVEL_TRACE);

    if (argc < 2) {
        printf("Usage: mf-window <app.mfapp> [--log-interval <seconds>]\n");
        return 1;
    }

    const char* mfapp_path = argv[1];
    mf_host_desc desc = {0};
    desc.log_interval = 5.0f;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--log-interval") == 0 && i + 1 < argc) {
            desc.log_interval = (float)atof(argv[++i]);
        }
    }

    if (mf_app_load_config(mfapp_path, &desc) != 0) {
        MF_LOG_ERROR("Failed to load application from %s", mfapp_path);
        return 1;
    }

    int result = mf_host_run(&desc);
    mf_host_desc_cleanup(&desc);

    return result;
}
