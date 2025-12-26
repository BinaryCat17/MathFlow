#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mathflow/isa/mf_base.h>
#include <mathflow/vm/mf_memory.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>

// Helper to read file
static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* buffer = malloc(length + 1);
    if (!buffer) return NULL;
    
    fread(buffer, 1, length, f);
    buffer[length] = '\0';
    fclose(f);
    return buffer;
}

static const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <path_to_graph.json | path_to_program.bin>\n", argv[0]);
        return 1;
    }

    const char* path = argv[1];
    printf("MathFlow Runner. Loading: %s\n", path);

    // 1. Setup Memory
    size_t arena_size = MF_MB(4);
    void* buffer = malloc(arena_size);
    mf_arena arena;
    mf_arena_init(&arena, buffer, arena_size);

    mf_program* prog = NULL;
    const char* ext = get_filename_ext(path);

    // 2. Load / Compile
    if (strcmp(ext, "json") == 0) {
        char* json = read_file(path);
        if (!json) { 
            printf("Error: Could not read file '%s'.\n", path);
            free(buffer);
            return 1; 
        }
        
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(json, &ir, &arena)) {
            printf("Error: Failed to parse JSON.\n");
            free(json);
            free(buffer);
            return 1;
        }
        printf("Graph Parsed: %zu nodes\n", ir.node_count);
        
        prog = mf_compile(&ir, &arena);
        free(json);
    } else if (strcmp(ext, "bin") == 0) {
        prog = mf_vm_load_program_from_file(path, &arena);
    } else {
        printf("Error: Unknown file extension '%s'. Use .json or .bin\n", ext);
        free(buffer);
        return 1;
    }

    if (!prog) {
        printf("Error: Failed to generate program.\n");
        free(buffer);
        return 1;
    }

    printf("Program Loaded: %u inst\n", prog->meta.instruction_count);

    // 3. Setup Backend & VM
    mf_backend_dispatch_table cpu_backend;
    mf_backend_cpu_init(&cpu_backend);
    
    mf_vm vm = {0};
    vm.backend = &cpu_backend;
    mf_vm_load_program(&vm, prog, &arena);
    
    // 4. Execute
    mf_vm_exec(&vm);
    
    // 5. Dump Output (Last element of each column)
    printf("\n--- Execution Finished ---\n");
    
    if (vm.f32_col && vm.f32_col->count > 0) {
        f32* v = (f32*)mf_column_get(vm.f32_col, vm.f32_col->count - 1);
        printf("[F32 Output]: %.4f\n", *v);
    }
    
    if (vm.vec2_col && vm.vec2_col->count > 0) {
        mf_vec2* v = (mf_vec2*)mf_column_get(vm.vec2_col, vm.vec2_col->count - 1);
        printf("[Vec2 Output]: {%.2f, %.2f}\n", v->x, v->y);
    }

    if (vm.vec3_col && vm.vec3_col->count > 0) {
        mf_vec3* v = (mf_vec3*)mf_column_get(vm.vec3_col, vm.vec3_col->count - 1);
        printf("[Vec3 Output]: {%.2f, %.2f, %.2f}\n", v->x, v->y, v->z);
    }

    if (vm.vec4_col && vm.vec4_col->count > 0) {
        mf_vec4* v = (mf_vec4*)mf_column_get(vm.vec4_col, vm.vec4_col->count - 1);
        printf("[Vec4 Output]: {%.2f, %.2f, %.2f, %.2f}\n", v->x, v->y, v->z, v->w);
    }

    if (vm.bool_col && vm.bool_col->count > 0) {
        u8* v = (u8*)mf_column_get(vm.bool_col, vm.bool_col->count - 1);
        printf("[Bool Output]: %s\n", *v ? "true" : "false");
    }

    free(buffer);
    return 0;
}