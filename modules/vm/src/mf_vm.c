#include <mathflow/vm/mf_vm.h>
#include <stdio.h>
#include <string.h>

void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena) {
    if (!vm || !prog || !arena) return;

    // 1. Setup Code
    vm->code = prog->code;
    vm->code_count = prog->meta.instruction_count;

    // 2. Setup Columns
    vm->f32_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->vec2_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->vec3_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->vec4_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->mat3_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->mat4_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->bool_col = MF_ARENA_PUSH(arena, mf_column, 1);

    // Initialize & Copy Data
    
    // F32
    mf_column_init(vm->f32_col, sizeof(f32), prog->meta.f32_count, arena);
    if (prog->meta.f32_count > 0 && prog->data_f32) {
        for (u32 i = 0; i < prog->meta.f32_count; ++i) {
            mf_column_push(vm->f32_col, &prog->data_f32[i], arena);
        }
    }

    // Vec2
    mf_column_init(vm->vec2_col, sizeof(mf_vec2), prog->meta.vec2_count, arena);
    if (prog->meta.vec2_count > 0 && prog->data_vec2) {
        for (u32 i = 0; i < prog->meta.vec2_count; ++i) {
            mf_column_push(vm->vec2_col, &prog->data_vec2[i], arena);
        }
    }
    
    // Vec3
    mf_column_init(vm->vec3_col, sizeof(mf_vec3), prog->meta.vec3_count, arena);
    if (prog->meta.vec3_count > 0 && prog->data_vec3) {
        for (u32 i = 0; i < prog->meta.vec3_count; ++i) {
            mf_column_push(vm->vec3_col, &prog->data_vec3[i], arena);
        }
    }

    // Vec4
    mf_column_init(vm->vec4_col, sizeof(mf_vec4), prog->meta.vec4_count, arena);
    if (prog->meta.vec4_count > 0 && prog->data_vec4) {
        for (u32 i = 0; i < prog->meta.vec4_count; ++i) {
            mf_column_push(vm->vec4_col, &prog->data_vec4[i], arena);
        }
    }

    // Mat3
    mf_column_init(vm->mat3_col, sizeof(mf_mat3), prog->meta.mat3_count, arena);
    if (prog->meta.mat3_count > 0 && prog->data_mat3) {
        for (u32 i = 0; i < prog->meta.mat3_count; ++i) {
            mf_column_push(vm->mat3_col, &prog->data_mat3[i], arena);
        }
    }

    // Mat4
    mf_column_init(vm->mat4_col, sizeof(mf_mat4), prog->meta.mat4_count, arena);
    if (prog->meta.mat4_count > 0 && prog->data_mat4) {
        for (u32 i = 0; i < prog->meta.mat4_count; ++i) {
            mf_column_push(vm->mat4_col, &prog->data_mat4[i], arena);
        }
    }

    // Bool (u8)
    mf_column_init(vm->bool_col, sizeof(u8), prog->meta.bool_count, arena);
    if (prog->meta.bool_count > 0 && prog->data_bool) {
        for (u32 i = 0; i < prog->meta.bool_count; ++i) {
            mf_column_push(vm->bool_col, &prog->data_bool[i], arena);
        }
    }
}

mf_program* mf_vm_load_program_from_file(const char* path, mf_arena* arena) {
    if (!path || !arena) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    mf_bin_header meta;
    if (fread(&meta, sizeof(mf_bin_header), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    
    if (meta.magic != MF_BINARY_MAGIC || meta.version != MF_BINARY_VERSION) {
        fclose(f);
        return NULL;
    }

    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta = meta;

    // Code
    prog->code = MF_ARENA_PUSH(arena, mf_instruction, meta.instruction_count);
    fread(prog->code, sizeof(mf_instruction), meta.instruction_count, f);

    // Data - Read in specific order!
    
    // 1. F32
    if (meta.f32_count > 0) {
        prog->data_f32 = MF_ARENA_PUSH(arena, f32, meta.f32_count);
        fread(prog->data_f32, sizeof(f32), meta.f32_count, f);
    } else prog->data_f32 = NULL;

    // 2. Vec2
    if (meta.vec2_count > 0) {
        prog->data_vec2 = MF_ARENA_PUSH(arena, mf_vec2, meta.vec2_count);
        fread(prog->data_vec2, sizeof(mf_vec2), meta.vec2_count, f);
    } else prog->data_vec2 = NULL;

    // 3. Vec3
    if (meta.vec3_count > 0) {
        prog->data_vec3 = MF_ARENA_PUSH(arena, mf_vec3, meta.vec3_count);
        fread(prog->data_vec3, sizeof(mf_vec3), meta.vec3_count, f);
    } else prog->data_vec3 = NULL;

    // 4. Vec4
    if (meta.vec4_count > 0) {
        prog->data_vec4 = MF_ARENA_PUSH(arena, mf_vec4, meta.vec4_count);
        fread(prog->data_vec4, sizeof(mf_vec4), meta.vec4_count, f);
    } else prog->data_vec4 = NULL;

    // 5. Mat3
    if (meta.mat3_count > 0) {
        prog->data_mat3 = MF_ARENA_PUSH(arena, mf_mat3, meta.mat3_count);
        fread(prog->data_mat3, sizeof(mf_mat3), meta.mat3_count, f);
    } else prog->data_mat3 = NULL;

    // 6. Mat4
    if (meta.mat4_count > 0) {
        prog->data_mat4 = MF_ARENA_PUSH(arena, mf_mat4, meta.mat4_count);
        fread(prog->data_mat4, sizeof(mf_mat4), meta.mat4_count, f);
    } else prog->data_mat4 = NULL;

    // 6. Bool
    if (meta.bool_count > 0) {
        prog->data_bool = MF_ARENA_PUSH(arena, u8, meta.bool_count);
        fread(prog->data_bool, sizeof(u8), meta.bool_count, f);
    } else prog->data_bool = NULL;

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

// --- Accessors ---

mf_ref_f32 mf_vm_map_f32(mf_vm* vm, u16 idx) {
    if (!vm || !vm->f32_col) return MF_NULL_F32;
    f32* p = (f32*)mf_column_get(vm->f32_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: F32[%u]\n", idx);
        return MF_NULL_F32;
    }
    return (mf_ref_f32){p};
}

mf_ref_vec2 mf_vm_map_vec2(mf_vm* vm, u16 idx) {
    if (!vm || !vm->vec2_col) return MF_NULL_VEC2;
    mf_vec2* p = (mf_vec2*)mf_column_get(vm->vec2_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Vec2[%u]\n", idx);
        return MF_NULL_VEC2;
    }
    return (mf_ref_vec2){p};
}

mf_ref_vec3 mf_vm_map_vec3(mf_vm* vm, u16 idx) {
    if (!vm || !vm->vec3_col) return MF_NULL_VEC3;
    mf_vec3* p = (mf_vec3*)mf_column_get(vm->vec3_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Vec3[%u]\n", idx);
        return MF_NULL_VEC3;
    }
    return (mf_ref_vec3){p};
}

mf_ref_vec4 mf_vm_map_vec4(mf_vm* vm, u16 idx) {
    if (!vm || !vm->vec4_col) return MF_NULL_VEC4;
    mf_vec4* p = (mf_vec4*)mf_column_get(vm->vec4_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Vec4[%u]\n", idx);
        return MF_NULL_VEC4;
    }
    return (mf_ref_vec4){p};
}

mf_ref_mat3 mf_vm_map_mat3(mf_vm* vm, u16 idx) {
    if (!vm || !vm->mat3_col) return MF_NULL_MAT3;
    mf_mat3* p = (mf_mat3*)mf_column_get(vm->mat3_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Mat3[%u]\n", idx);
        return MF_NULL_MAT3;
    }
    return (mf_ref_mat3){p};
}

mf_ref_mat4 mf_vm_map_mat4(mf_vm* vm, u16 idx) {
    if (!vm || !vm->mat4_col) return MF_NULL_MAT4;
    mf_mat4* p = (mf_mat4*)mf_column_get(vm->mat4_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Mat4[%u]\n", idx);
        return MF_NULL_MAT4;
    }
    return (mf_ref_mat4){p};
}

mf_ref_bool mf_vm_map_bool(mf_vm* vm, u16 idx) {
    if (!vm || !vm->bool_col) return MF_NULL_BOOL;
    u8* p = (u8*)mf_column_get(vm->bool_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Bool[%u]\n", idx);
        return MF_NULL_BOOL;
    }
    return (mf_ref_bool){p};
}

// --- Counts ---

size_t mf_vm_get_count_f32(mf_vm* vm) { return (vm && vm->f32_col) ? vm->f32_col->count : 0; }
size_t mf_vm_get_count_vec2(mf_vm* vm) { return (vm && vm->vec2_col) ? vm->vec2_col->count : 0; }
size_t mf_vm_get_count_vec3(mf_vm* vm) { return (vm && vm->vec3_col) ? vm->vec3_col->count : 0; }
size_t mf_vm_get_count_vec4(mf_vm* vm) { return (vm && vm->vec4_col) ? vm->vec4_col->count : 0; }
size_t mf_vm_get_count_mat3(mf_vm* vm) { return (vm && vm->mat3_col) ? vm->mat3_col->count : 0; }
size_t mf_vm_get_count_mat4(mf_vm* vm) { return (vm && vm->mat4_col) ? vm->mat4_col->count : 0; }
size_t mf_vm_get_count_bool(mf_vm* vm) { return (vm && vm->bool_col) ? vm->bool_col->count : 0; }
