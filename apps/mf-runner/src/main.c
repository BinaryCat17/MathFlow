#include <stdio.h>
#include <stdlib.h>
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
    fread(buffer, 1, length, f);
    buffer[length] = '\0';
    fclose(f);
    return buffer;
}

int main(void) {
    printf("MathFlow Engine Initialized (Modular)\n");

    // 1. Setup Memory
    size_t arena_size = MF_MB(1);
    void* buffer = malloc(arena_size);
    mf_arena arena;
    mf_arena_init(&arena, buffer, arena_size);

    // 2. Load Graph & Compile
    const char* path = "assets/graphs/simple_math.json";
    char* json = read_file(path);
    if (!json) {
         // Fallback for different CWD
         json = read_file("../../../assets/graphs/simple_math.json");
    }
    
    if (!json) {
        printf("Failed to read file: %s\n", path);
        return 1;
    }
    printf("Graph loaded.\n");

    mf_graph_ir ir = {0};
    if (!mf_compile_load_json(json, &ir, &arena)) {
        printf("Failed to parse JSON\n");
        return 1;
    }
    printf("Graph Parsed: %zu nodes, %zu links\n", ir.node_count, ir.link_count);

    // 3. Compile to Program (Decoupled)
    mf_program* prog = mf_compile(&ir, &arena);
    printf("Compiled: %u instructions\n", prog->meta.instruction_count);

    // --- I/O Test ---
    const char* bin_path = "test_prog.bin";
    if (mf_compile_save_program(prog, bin_path)) {
        printf("Saved program to %s\n", bin_path);
    } else {
        printf("Failed to save program to %s\n", bin_path);
    }

    mf_program* prog2 = mf_vm_load_program_from_file(bin_path, &arena);
    if (prog2) {
        printf("Loaded program from %s (Inst: %u)\n", bin_path, prog2->meta.instruction_count);
        // Use the loaded program to verify IO
        prog = prog2;
    } else {
        printf("Failed to load program from %s\n", bin_path);
    }
    // ----------------

    // 4. Setup Backend (CPU)
    mf_backend_dispatch_table cpu_backend;
    mf_backend_cpu_init(&cpu_backend);

    // 5. Setup VM
    mf_vm vm = {0};
    vm.backend = &cpu_backend;

    // 6. Load Program into VM
    mf_vm_load_program(&vm, prog, &arena);
    printf("Program Loaded. Memory initialized.\n");

    // 7. Execute
    mf_vm_exec(&vm);

    // 8. Verify Results
    // We expect the result at index 3 of vec3 column (based on simple_math.json structure)
    if (vm.vec3_col && vm.vec3_col->count > 3) {
        mf_vec3* res = (mf_vec3*)mf_column_get(vm.vec3_col, 3);
        if (res) {
            printf("Final Result: {%.2f, %.2f, %.2f}\n", res->x, res->y, res->z);
            printf("Expected:     {10.00, 14.00, 18.00}\n");
        }
    } else {
        printf("Error: Could not retrieve result (index 3 out of bounds).\n");
    }

    free(json);
    free(buffer);
    return 0;
}