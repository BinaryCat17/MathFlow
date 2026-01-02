#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/host/mf_host_desc.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_platform.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv) {
    mf_host_init_logger();
    mf_log_set_global_level(MF_LOG_LEVEL_INFO);

    if (argc < 2) {
        printf("Usage: mf-window <app.mfapp> [--log-interval <seconds>] [--trace] [--debug]\n");
        return 1;
    }

    const char* mfapp_path = argv[1];
    mf_host_desc desc = {0};
    desc.log_interval = 5.0f;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--log-interval") == 0 && i + 1 < argc) {
            desc.log_interval = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--trace") == 0) {
            mf_log_set_global_level(MF_LOG_LEVEL_TRACE);
        } else if (strcmp(argv[i], "--debug") == 0) {
            mf_log_set_global_level(MF_LOG_LEVEL_DEBUG);
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
