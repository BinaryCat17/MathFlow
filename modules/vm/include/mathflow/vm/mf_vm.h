#ifndef MF_VM_H
#define MF_VM_H

#include <mathflow/isa/mf_instruction.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/vm/mf_memory.h>

// Forward decl
struct mf_vm_t;

// --- Backend Interface ---
// The VM doesn't know math. It calls these functions.
// We pass the VM state + indices so the backend can lookup data in columns.
typedef void (*mf_op_func)(struct mf_vm_t* vm, u16 dest, u16 src1, u16 src2);

typedef struct {
    mf_op_func op_table[MF_OP_COUNT];
} mf_backend_dispatch_table;

// --- VM State ---
typedef struct mf_vm_t {
    mf_instruction* code;
    size_t code_count;
    
    // Data Columns (The "Matrix")
    mf_column* f32_col;
    mf_column* vec2_col;
    mf_column* vec3_col;
    mf_column* vec4_col;
    mf_column* mat4_col;
    mf_column* bool_col;

    // Backend dispatch
    const mf_backend_dispatch_table* backend;
} mf_vm;

// Load a program into the VM (Allocates columns and copies initial data)
// Requires an arena for internal column structures.
void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena);

// Load a program from a .bin file into the arena
mf_program* mf_vm_load_program_from_file(const char* path, mf_arena* arena);

void mf_vm_exec(mf_vm* vm);

#endif // MF_VM_H
