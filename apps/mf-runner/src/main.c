#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/vm/mf_vm.h>

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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <path_to_graph.json | path_to_program.bin>\n", argv[0]);
        return 1;
    }

    const char* path = argv[1];
    int frames = 1;
    if (argc >= 4 && strcmp(argv[2], "--frames") == 0) {
        frames = atoi(argv[3]);
    }

    printf("MathFlow Tensor Runner. Loading: %s\n", path);

    // 1. Initialize Engine (Code, Constants, Backend)
    mf_engine engine;
    mf_engine_init(&engine, NULL);

    if (!mf_engine_load_graph(&engine, path)) {
        printf("Error: Failed to load graph.\n");
        mf_engine_shutdown(&engine);
        return 1;
    }

    printf("Program: %u tensors, %u insts\n", engine.program->meta.tensor_count, engine.program->meta.instruction_count);

    // 2. Setup Execution State (Heap, VM)
    // Heap for Tensor Data (Dynamic)
    size_t heap_size = MF_MB(64); 
    void* heap_buffer = malloc(heap_size); 
    mf_heap heap;
    mf_heap_init(&heap, heap_buffer, heap_size);

    mf_vm vm;
    mf_vm_init(&vm, &engine.ctx, (mf_allocator*)&heap);
    
    // Alloc registers & Memory Nodes
    mf_vm_reset(&vm, &engine.arena);
    
    // 3. Execute
    printf("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        mf_vm_exec(&vm);
        
        // Debug output for first few frames
        if (f < 5) {
             printf("Frame %d:\n", f);
             for(u32 i=0; i<vm.register_count; ++i) {
                mf_tensor* t = &vm.registers[i];
                
                // Find name
                const char* name = NULL;
                for (u32 s = 0; s < vm.ctx->symbol_count; ++s) {
                    if (vm.ctx->symbols[s].register_idx == i) {
                        name = vm.ctx->symbols[s].name;
                        break;
                    }
                }

                if (engine.program->meta.tensor_count < 20) {
                    print_tensor(i, name, t);
                }
             }
        }
    }
    
    // 4. Dump All Tensors
    printf("\n--- Execution Finished ---\n");
    for(u32 i=0; i<vm.register_count; ++i) {
        mf_tensor* t = mf_vm_map_tensor(&vm, i, MF_ACCESS_READ);

        const char* name = NULL;
        for (u32 s = 0; s < vm.ctx->symbol_count; ++s) {
            if (vm.ctx->symbols[s].register_idx == i) {
                name = vm.ctx->symbols[s].name;
                break;
            }
        }
        print_tensor(i, name, t);
    }

    // 5. Cleanup
    printf("\n[Memory Stats] Used: %zu, Peak: %zu, Allocations: %zu\n", 
           heap.used_memory, heap.peak_memory, heap.allocation_count);

    mf_vm_shutdown(&vm);
    free(heap_buffer);
    
    mf_engine_shutdown(&engine);
    return 0;
}