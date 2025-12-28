#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/host/mf_manifest_loader.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "assets/apps/sdf_button.mfapp";
    
    // Check extension
    const char* ext = strrchr(path, '.');
    if (!ext || strcmp(ext, ".mfapp") != 0) {
        printf("Error: MathFlow Host only accepts .mfapp manifests.\n");
        printf("Usage: %s <app.mfapp>\n", argv[0]);
        return 1;
    }

    mf_host_desc desc = {0};
    int res = mf_app_load_config(path, &desc);
    if (res != 0) {
        printf("Failed to load manifest: %s (Error code: %d)\n", path, res);
        return 1;
    }
    
    printf("Starting App: %s\nMode: %s\nGraph: %s\n", 
        desc.window_title, 
        desc.runtime_type == MF_HOST_RUNTIME_SHADER ? "Shader (GPU-like)" : "Script (CPU-like)",
        desc.graph_path
    );

    return mf_host_run(&desc);
}
