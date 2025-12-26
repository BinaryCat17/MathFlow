#include <mathflow/vm/mf_vm.h>
#include <stdio.h>
#include <string.h>

void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena) {
    if (!vm || !prog || !arena) return;

    // 1. Setup Code
    vm->code = prog->code;
    vm->code_count = prog->meta.instruction_count;

    // 2. Setup Columns
    // Allocate column structures in arena
    vm->f32_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->vec3_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->mat4_col = MF_ARENA_PUSH(arena, mf_column, 1);

    // Initialize columns with initial data from program
    
    // F32
    // We allocate enough capacity for the initial data + some extra if needed (but here we just load)
    mf_column_init(vm->f32_col, sizeof(f32), prog->meta.f32_count, arena);
    if (prog->meta.f32_count > 0 && prog->data_f32) {
        for (u32 i = 0; i < prog->meta.f32_count; ++i) {
            mf_column_push(vm->f32_col, &prog->data_f32[i], arena);
        }
    }
    
    // Vec3
    mf_column_init(vm->vec3_col, sizeof(mf_vec3), prog->meta.vec3_count, arena);
    if (prog->meta.vec3_count > 0 && prog->data_vec3) {
        for (u32 i = 0; i < prog->meta.vec3_count; ++i) {
            mf_column_push(vm->vec3_col, &prog->data_vec3[i], arena);
        }
    }

    // Mat4
    mf_column_init(vm->mat4_col, sizeof(mf_mat4), prog->meta.mat4_count, arena);
    if (prog->meta.mat4_count > 0 && prog->data_mat4) {
        for (u32 i = 0; i < prog->meta.mat4_count; ++i) {
            mf_column_push(vm->mat4_col, &prog->data_mat4[i], arena);
        }
    }
}

mf_program* mf_vm_load_program_from_file(const char* path, mf_arena* arena) {
    if (!path || !arena) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    // Read Header
    mf_bin_header meta;
    if (fread(&meta, sizeof(mf_bin_header), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    
    if (meta.magic != MF_BINARY_MAGIC || meta.version != MF_BINARY_VERSION) {
        fclose(f);
        return NULL;
    }

    // Allocate Program Structure
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta = meta;

    // Allocate & Read Code
    prog->code = MF_ARENA_PUSH(arena, mf_instruction, meta.instruction_count);
    fread(prog->code, sizeof(mf_instruction), meta.instruction_count, f);

    // Allocate & Read Data
    if (meta.f32_count > 0) {
        prog->data_f32 = MF_ARENA_PUSH(arena, f32, meta.f32_count);
        fread(prog->data_f32, sizeof(f32), meta.f32_count, f);
    } else {
        prog->data_f32 = NULL;
    }

    if (meta.vec3_count > 0) {
        prog->data_vec3 = MF_ARENA_PUSH(arena, mf_vec3, meta.vec3_count);
        fread(prog->data_vec3, sizeof(mf_vec3), meta.vec3_count, f);
    } else {
        prog->data_vec3 = NULL;
    }

    if (meta.mat4_count > 0) {
        prog->data_mat4 = MF_ARENA_PUSH(arena, mf_mat4, meta.mat4_count);
        fread(prog->data_mat4, sizeof(mf_mat4), meta.mat4_count, f);
    } else {
        prog->data_mat4 = NULL;
    }

    fclose(f);
    return prog;
}

void mf_vm_exec(mf_vm* vm) {
    if (!vm->backend) return;

    for (size_t i = 0; i < vm->code_count; ++i) {
        mf_instruction inst = vm->code[i];
        
        if (inst.opcode < MF_OP_COUNT) {
            mf_op_func func = vm->backend->op_table[inst.opcode];
            if (func) {
                func(vm, inst.dest_idx, inst.src1_idx, inst.src2_idx);
            }
        }
    }
}