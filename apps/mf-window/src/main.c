#include <mathflow/host/mf_host.h>
#include <stdio.h>

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "assets/graphs/sdf_button.json";
    
    mf_host_desc desc = {
        .window_title = "MathFlow Visualizer",
        .width = 800,
        .height = 600,
        .graph_path = path,
        .num_threads = 0 // Auto
    };
    
    return mf_host_run(&desc);
}