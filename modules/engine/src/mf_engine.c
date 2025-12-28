#include <mathflow/engine/mf_engine.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/vm/mf_vm_utils.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

void mf_engine_init(mf_engine* engine, const mf_engine_desc* desc) {
    memset(engine, 0, sizeof(mf_engine));
    
    size_t size = (desc && desc->arena_size > 0) ? desc->arena_size : MF_MB(8);
    
    engine->arena_buffer = malloc(size);
    mf_arena_init(&engine->arena, engine->arena_buffer, size);
    
    // Initialize default CPU backend
    // In the future, this could be configurable (GPU/Metal/Vulkan)
    mf_backend_cpu_init(&engine->backend);
}

bool mf_engine_load_graph(mf_engine* engine, const char* path) {
    if (!engine->arena_buffer) return false;
    
    const char* ext = get_filename_ext(path);
    mf_program* prog = NULL;

    if (strcmp(ext, "json") == 0) {
        // Compile from source
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(path, &ir, &engine->arena)) {
            printf("[Engine] Failed to load/parse JSON: %s\n", path);
            return false;
        }
        
        prog = mf_compile(&ir, &engine->arena);
    } else if (strcmp(ext, "bin") == 0) {
        // Load binary
        prog = mf_vm_load_program_from_file(path, &engine->arena);
    } else {
        printf("[Engine] Unknown file extension: %s\n", ext);
        return false;
    }

    if (!prog) {
        printf("[Engine] Failed to generate program.\n");
        return false;
    }

    engine->program = prog;
    
    // Setup Context
    mf_context_init(&engine->ctx, engine->program, &engine->backend);
    
    return true;
}

void mf_engine_shutdown(mf_engine* engine) {
    if (engine->arena_buffer) {
        free(engine->arena_buffer);
        engine->arena_buffer = NULL;
    }
    memset(engine, 0, sizeof(mf_engine));
}
