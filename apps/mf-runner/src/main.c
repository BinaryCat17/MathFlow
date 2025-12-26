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
    printf("MathFlow Logic Test\n");

    // 1. Setup Memory
    size_t arena_size = MF_MB(1);
    void* buffer = malloc(arena_size);
    mf_arena arena;
    mf_arena_init(&arena, buffer, arena_size);

    // 2. Load Graph
    const char* path = "assets/graphs/logic_test.json";
    char* json = read_file(path);
    if (!json) {
         json = read_file("../../../assets/graphs/logic_test.json");
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

    // 3. Compile
    mf_program* prog = mf_compile(&ir, &arena);
    printf("Compiled: %u instructions\n", prog->meta.instruction_count);

    // --- I/O Test (Verify binary format with new types) ---
    const char* bin_path = "logic_test.bin";
    if (mf_compile_save_program(prog, bin_path)) {
        printf("Saved program to %s\n", bin_path);
    }
    
    mf_program* prog2 = mf_vm_load_program_from_file(bin_path, &arena);
    if (prog2) {
        printf("Loaded program from %s\n", bin_path);
        prog = prog2;
    }
    // ----------------

    // 4. Setup Backend
    mf_backend_dispatch_table cpu_backend;
    mf_backend_cpu_init(&cpu_backend);

    // 5. Setup VM
    mf_vm vm = {0};
    vm.backend = &cpu_backend;

    mf_vm_load_program(&vm, prog, &arena);
    printf("Program Loaded. Memory initialized.\n");

    // 6. Execute
    mf_vm_exec(&vm);

    // 7. Verify Results (Expect Green: 0, 1, 0, 1)
    // The output node is ID 6. It's the 3rd Vec4 node (ID 4, 5 are inputs).
    // Or just check the last element.
    if (vm.vec4_col && vm.vec4_col->count > 0) {
        size_t idx = vm.vec4_col->count - 1;
        mf_vec4* res = (mf_vec4*)mf_column_get(vm.vec4_col, idx);
        if (res) {
            printf("Final Result: {%.2f, %.2f, %.2f, %.2f}\n", res->x, res->y, res->z, res->w);
            printf("Expected:     {0.00, 1.00, 0.00, 1.00}\n");
            
            if (res->y == 1.0f && res->x == 0.0f) {
                printf("SUCCESS: Logic Correct.\n");
            } else {
                printf("FAILURE: Logic Incorrect.\n");
            }
        }
    } else {
        printf("Error: No Vec4 output.\n");
    }

    free(json);
    free(buffer);
    return 0;
}
