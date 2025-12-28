#include <mathflow/host/mf_host_headless.h>
#include <mathflow/vm/mf_vm.h>
#include <stdio.h>
#include <string.h>

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

int mf_host_run_headless(mf_engine* engine, int frames) {
    if (!engine || !engine->program) return 1;

    // Create Instance (VM + Heap)
    mf_instance inst;
    if (!mf_engine_create_instance(engine, &inst, 0)) { // Default heap
        printf("Error: Failed to create VM instance.\n");
        return 1;
    }

    // Execute
    printf("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        mf_vm_exec(&inst.vm);
        
        if (inst.vm.error != MF_ERROR_NONE) {
            printf("Error: Runtime error %d\n", inst.vm.error);
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
                if (engine->program->meta.tensor_count < 20 || (name && strncmp(name, "out_", 4) == 0)) {
                    print_tensor(i, name, t);
                }
             }
        }
    }
    
    // Dump Final State
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

    printf("\n[Memory Stats] Used: %zu, Peak: %zu, Allocations: %zu\n", 
           inst.heap.used_memory, inst.heap.peak_memory, inst.heap.allocation_count);

    mf_instance_destroy(&inst);
    return 0;
}
