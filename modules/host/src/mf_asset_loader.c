#include <mathflow/host/mf_asset_loader.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/engine/mf_engine.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

bool mf_asset_loader_load(mf_engine* engine, const char* path) {
    mf_arena* arena = mf_engine_get_arena(engine);
    if (!arena) return false;
    
    const char* ext = get_filename_ext(path);
    mf_program* prog = NULL;

    if (strcmp(ext, "json") == 0) {
        // Compile from source
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(path, &ir, arena)) {
            printf("[AssetLoader] Failed to load/parse JSON: %s\n", path);
            return false;
        }
        
        prog = mf_compile(&ir, arena);
    } else if (strcmp(ext, "bin") == 0) {
        // Load binary
        prog = mf_vm_load_program_from_file(path, arena);
    } else {
        printf("[AssetLoader] Unknown file extension: %s\n", ext);
        return false;
    }

    if (!prog) {
        printf("[AssetLoader] Failed to generate program.\n");
        return false;
    }

    // Bind program to engine
    // Since we don't have a public API for this yet in the engine refactor step,
    // we access the struct directly for now, OR we add the API to engine first.
    // Ideally, we should add mf_engine_bind_program in the next step.
    // For now, I'll assume direct access or use the function I'm about to add.
    
    // Use direct access for this intermediate step, will be cleaned up when engine is updated.
    // Actually, I'll update engine first in the next tool call, but I'm writing this file now.
    // I will use a hypothetical mf_engine_bind_program function and implement it in engine next.
    mf_engine_bind_program(engine, prog);
    
    return true;
}
