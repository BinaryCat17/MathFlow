#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/host/mf_app_loader.h>

static void print_tensor(u32 idx, const char* name, mf_tensor* t) {
    if (!t || !t->data) {
        printf("  [%u]%s: (Empty)\n", idx, name ? name : "");
        return;
    }
    
    printf("  [%u] ", idx);
    if (name) printf("'%s' ", name); 
    
    // Print Shape
    printf("Shape: [");
    for(int i=0; i<t->ndim; ++i) printf("%d%s", t->shape[i], i < t->ndim-1 ? "," : "");
    printf("] ");
    
    // Print Data
    if (t->dtype == MF_DTYPE_F32) {
        printf("F32: {");
        f32* data = (f32*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) {
            printf("%.2f%s", data[i], i < limit-1 ? ", " : "");
        }
        if (t->size > limit) printf("... (+%zu)", t->size - limit);
        printf("}\n");
    } else if (t->dtype == MF_DTYPE_I32) {
        printf("I32: {");
        int32_t* data = (int32_t*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) {
            printf("%d%s", data[i], i < limit-1 ? ", " : "");
        }
        if (t->size > limit) printf("... (+%zu)", t->size - limit);
        printf("}\n");
    } else if (t->dtype == MF_DTYPE_U8) {
        printf("Bool: {");
        u8* data = (u8*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) {
            printf("%s%s", data[i] ? "true" : "false", i < limit-1 ? ", " : "");
        }
        printf("}\n");
    }
}

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

    // 1. Load Manifest
    mf_host_desc app_desc = {0};
    if (mf_app_load_config(mfapp_path, &app_desc) != 0) {
        // Fallback: Try loading as raw graph if not mfapp
        // This keeps backward compatibility for quick testing
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
    printf("Mode: %s\n", app_desc.runtime_type == MF_HOST_RUNTIME_SHADER ? "Shader (Simulated)" : "Script");

    // 2. Initialize Engine
    mf_engine engine;
    mf_engine_init(&engine, NULL);

    if (!mf_engine_load_graph(&engine, app_desc.graph_path)) {
        printf("Error: Failed to load graph.\n");
        mf_engine_shutdown(&engine);
        return 1;
    }

    printf("Program: %u tensors, %u insts\n", engine.program->meta.tensor_count, engine.program->meta.instruction_count);

    // 3. Create Instance (VM + Heap)
    mf_instance inst;
    if (!mf_engine_create_instance(&engine, &inst, 0)) { // Default heap
        printf("Error: Failed to create VM instance.\n");
        mf_engine_shutdown(&engine);
        return 1;
    }

    // 4. Execute
    printf("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        mf_vm_exec(&inst.vm);
        
        if (inst.vm.error != MF_ERROR_NONE) {
            printf("Error: Runtime error %d at instruction %zu\n", inst.vm.error, (size_t)0); // PC not tracked yet
            break;
        }
        
        // Debug output for first few frames
        if (f < 3) {
             printf("--- Frame %d ---\n", f);
             for(u32 i=0; i<inst.vm.register_count; ++i) {
                mf_tensor* t = &inst.vm.registers[i];
                
                // Find name
                const char* name = NULL;
                for (u32 s = 0; s < inst.vm.ctx->symbol_count; ++s) {
                    if (inst.vm.ctx->symbols[s].register_idx == i) {
                        name = inst.vm.ctx->symbols[s].name;
                        break;
                    }
                }
                
                // Print only interesting tensors (small program) or outputs
                if (engine.program->meta.tensor_count < 20 || (name && strncmp(name, "out_", 4) == 0)) {
                    print_tensor(i, name, t);
                }
             }
        }
    }
    
    // 5. Dump Final State
    printf("\n--- Final State ---\n");
    for(u32 i=0; i<inst.vm.register_count; ++i) {
        mf_tensor* t = mf_vm_map_tensor(&inst.vm, i, MF_ACCESS_READ);

        const char* name = NULL;
        for (u32 s = 0; s < inst.vm.ctx->symbol_count; ++s) {
            if (inst.vm.ctx->symbols[s].register_idx == i) {
                name = inst.vm.ctx->symbols[s].name;
                break;
            }
        }
        print_tensor(i, name, t);
    }

    // 6. Cleanup
    printf("\n[Memory Stats] Used: %zu, Peak: %zu, Allocations: %zu\n", 
           inst.heap.used_memory, inst.heap.peak_memory, inst.heap.allocation_count);

    mf_instance_destroy(&inst);
    mf_engine_shutdown(&engine);
    
    // Free desc strings if allocated by loader (strdup)
    // Currently mf_app_loader implementation does allocate strings, but we exit anyway.
    // For correctness we should free, but it's a short-lived CLI.
    
    return 0;
}
