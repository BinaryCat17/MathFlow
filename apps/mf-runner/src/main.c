#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>

static const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

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

    // 1. Setup Memory
    // Arena for Program Code & Metadata
    size_t arena_size = MF_MB(8); 
    void* arena_buffer = malloc(arena_size);
    mf_arena arena;
    mf_arena_init(&arena, arena_buffer, arena_size);

    // Heap for Tensor Data (Dynamic)
    size_t heap_size = MF_MB(64); 
    void* heap_buffer = malloc(heap_size);
    mf_heap heap;
    mf_heap_init(&heap, heap_buffer, heap_size);

    mf_program* prog = NULL;
    const char* ext = get_filename_ext(path);

    // 2. Load / Compile
    if (strcmp(ext, "json") == 0) {
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(path, &ir, &arena)) {
            printf("Error: Failed to parse JSON or expand graph.\n");
            free(arena_buffer); free(heap_buffer); return 1;
        }
        
        prog = mf_compile(&ir, &arena);
    } else if (strcmp(ext, "bin") == 0) {
        prog = mf_vm_load_program_from_file(path, &arena);
    }

    if (!prog) {
        printf("Error: Failed to generate program.\n");
        free(arena_buffer); free(heap_buffer); return 1;
    }

    printf("Program: %u tensors, %u insts\n", prog->meta.tensor_count, prog->meta.instruction_count);

    // 3. Setup Backend & Context & VM
    mf_backend_dispatch_table cpu_backend;
    mf_backend_cpu_init(&cpu_backend);
    
    mf_context ctx;
    mf_context_init(&ctx, prog, &cpu_backend);

    mf_vm vm;
    mf_vm_init(&vm, &ctx, (mf_allocator*)&heap);
    
    // Alloc registers
    mf_vm_reset(&vm, &arena);
    
    // 4. Execute
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

                // Heuristic to find 'interesting' tensors (non-constant inputs)
                // For now just dump everything for small graphs
                if (prog->meta.tensor_count < 20) {
                    print_tensor(i, name, t);
                }
             }
        }
    }
    
    // 5. Dump All Tensors
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

    // 6. Cleanup
    printf("\n[Memory Stats] Used: %zu, Peak: %zu, Allocations: %zu\n", 
           heap.used_memory, heap.peak_memory, heap.allocation_count);

    mf_vm_shutdown(&vm);

    free(arena_buffer);
    free(heap_buffer);
    return 0;
}
