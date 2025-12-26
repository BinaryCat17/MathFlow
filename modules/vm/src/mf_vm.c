#include <mathflow/vm/mf_vm.h>
#include <stdio.h>
#include <string.h>

void mf_vm_load_program(mf_vm* vm, const mf_program* prog, mf_arena* arena) {
    if (!vm || !prog || !arena) return;

    // 1. Setup Code
    vm->_code = prog->code;
    vm->_code_count = prog->meta.instruction_count;

    // 2. Setup Columns
    vm->_f32_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->_vec2_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->_vec3_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->_vec4_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->_mat3_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->_mat4_col = MF_ARENA_PUSH(arena, mf_column, 1);
    vm->_bool_col = MF_ARENA_PUSH(arena, mf_column, 1);

    // Initialize & Copy Data
    
    // F32
    mf_column_init(vm->_f32_col, sizeof(f32), prog->meta.f32_count, arena);
    if (prog->meta.f32_count > 0 && prog->data_f32) {
        for (u32 i = 0; i < prog->meta.f32_count; ++i) {
            mf_column_push(vm->_f32_col, &prog->data_f32[i], arena);
        }
    }

    // Vec2
    mf_column_init(vm->_vec2_col, sizeof(mf_vec2), prog->meta.vec2_count, arena);
    if (prog->meta.vec2_count > 0 && prog->data_vec2) {
        for (u32 i = 0; i < prog->meta.vec2_count; ++i) {
            mf_column_push(vm->_vec2_col, &prog->data_vec2[i], arena);
        }
    }
    
    // Vec3
    mf_column_init(vm->_vec3_col, sizeof(mf_vec3), prog->meta.vec3_count, arena);
    if (prog->meta.vec3_count > 0 && prog->data_vec3) {
        for (u32 i = 0; i < prog->meta.vec3_count; ++i) {
            mf_column_push(vm->_vec3_col, &prog->data_vec3[i], arena);
        }
    }

    // Vec4
    mf_column_init(vm->_vec4_col, sizeof(mf_vec4), prog->meta.vec4_count, arena);
    if (prog->meta.vec4_count > 0 && prog->data_vec4) {
        for (u32 i = 0; i < prog->meta.vec4_count; ++i) {
            mf_column_push(vm->_vec4_col, &prog->data_vec4[i], arena);
        }
    }

    // Mat3
    mf_column_init(vm->_mat3_col, sizeof(mf_mat3), prog->meta.mat3_count, arena);
    if (prog->meta.mat3_count > 0 && prog->data_mat3) {
        for (u32 i = 0; i < prog->meta.mat3_count; ++i) {
            mf_column_push(vm->_mat3_col, &prog->data_mat3[i], arena);
        }
    }

    // Mat4
    mf_column_init(vm->_mat4_col, sizeof(mf_mat4), prog->meta.mat4_count, arena);
    if (prog->meta.mat4_count > 0 && prog->data_mat4) {
        for (u32 i = 0; i < prog->meta.mat4_count; ++i) {
            mf_column_push(vm->_mat4_col, &prog->data_mat4[i], arena);
        }
    }

    // Bool (u8)
    mf_column_init(vm->_bool_col, sizeof(u8), prog->meta.bool_count, arena);
    if (prog->meta.bool_count > 0 && prog->data_bool) {
        for (u32 i = 0; i < prog->meta.bool_count; ++i) {
            mf_column_push(vm->_bool_col, &prog->data_bool[i], arena);
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
    if (!vm || !vm->backend) return;

    if (vm->backend->on_exec_begin) {
        vm->backend->on_exec_begin(vm);
    }

    for (size_t i = 0; i < vm->_code_count; ++i) {
        mf_instruction inst = vm->_code[i];
        
        if (inst.opcode < MF_OP_COUNT) {
            mf_op_func func = vm->backend->op_table[inst.opcode];
            if (func) {
                func(vm, inst.dest_idx, inst.src1_idx, inst.src2_idx);
            }
        }
    }

    if (vm->backend->on_exec_end) {
        vm->backend->on_exec_end(vm);
    }
}

// --- Accessors ---

mf_ref_f32 mf_vm_map_f32(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (!vm || !vm->_f32_col) return MF_NULL_F32;
    if (vm->backend && vm->backend->on_map) vm->backend->on_map(vm, MF_COL_F32, idx, mode);
    
    f32* p = (f32*)mf_column_get(vm->_f32_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: F32[%u]\n", idx);
        return MF_NULL_F32;
    }
    return (mf_ref_f32){p};
}

mf_ref_vec2 mf_vm_map_vec2(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (!vm || !vm->_vec2_col) return MF_NULL_VEC2;
    if (vm->backend && vm->backend->on_map) vm->backend->on_map(vm, MF_COL_VEC2, idx, mode);

    mf_vec2* p = (mf_vec2*)mf_column_get(vm->_vec2_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Vec2[%u]\n", idx);
        return MF_NULL_VEC2;
    }
    return (mf_ref_vec2){p};
}

mf_ref_vec3 mf_vm_map_vec3(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (!vm || !vm->_vec3_col) return MF_NULL_VEC3;
    if (vm->backend && vm->backend->on_map) vm->backend->on_map(vm, MF_COL_VEC3, idx, mode);

    mf_vec3* p = (mf_vec3*)mf_column_get(vm->_vec3_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Vec3[%u]\n", idx);
        return MF_NULL_VEC3;
    }
    return (mf_ref_vec3){p};
}

mf_ref_vec4 mf_vm_map_vec4(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (!vm || !vm->_vec4_col) return MF_NULL_VEC4;
    if (vm->backend && vm->backend->on_map) vm->backend->on_map(vm, MF_COL_VEC4, idx, mode);

    mf_vec4* p = (mf_vec4*)mf_column_get(vm->_vec4_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Vec4[%u]\n", idx);
        return MF_NULL_VEC4;
    }
    return (mf_ref_vec4){p};
}

mf_ref_mat3 mf_vm_map_mat3(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (!vm || !vm->_mat3_col) return MF_NULL_MAT3;
    if (vm->backend && vm->backend->on_map) vm->backend->on_map(vm, MF_COL_MAT3, idx, mode);

    mf_mat3* p = (mf_mat3*)mf_column_get(vm->_mat3_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Mat3[%u]\n", idx);
        return MF_NULL_MAT3;
    }
    return (mf_ref_mat3){p};
}

mf_ref_mat4 mf_vm_map_mat4(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (!vm || !vm->_mat4_col) return MF_NULL_MAT4;
    if (vm->backend && vm->backend->on_map) vm->backend->on_map(vm, MF_COL_MAT4, idx, mode);

    mf_mat4* p = (mf_mat4*)mf_column_get(vm->_mat4_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Mat4[%u]\n", idx);
        return MF_NULL_MAT4;
    }
    return (mf_ref_mat4){p};
}

mf_ref_bool mf_vm_map_bool(mf_vm* vm, u16 idx, mf_access_mode mode) {
    if (!vm || !vm->_bool_col) return MF_NULL_BOOL;
    if (vm->backend && vm->backend->on_map) vm->backend->on_map(vm, MF_COL_BOOL, idx, mode);

    u8* p = (u8*)mf_column_get(vm->_bool_col, idx);
    if (!p) {
        fprintf(stderr, "[VM Error] Index Out of Bounds: Bool[%u]\n", idx);
        return MF_NULL_BOOL;
    }
    return (mf_ref_bool){p};
}

// --- Counts ---

size_t mf_vm_get_count_f32(mf_vm* vm) { return (vm && vm->_f32_col) ? vm->_f32_col->count : 0; }
size_t mf_vm_get_count_vec2(mf_vm* vm) { return (vm && vm->_vec2_col) ? vm->_vec2_col->count : 0; }
size_t mf_vm_get_count_vec3(mf_vm* vm) { return (vm && vm->_vec3_col) ? vm->_vec3_col->count : 0; }
size_t mf_vm_get_count_vec4(mf_vm* vm) { return (vm && vm->_vec4_col) ? vm->_vec4_col->count : 0; }
size_t mf_vm_get_count_mat3(mf_vm* vm) { return (vm && vm->_mat3_col) ? vm->_mat3_col->count : 0; }
size_t mf_vm_get_count_mat4(mf_vm* vm) { return (vm && vm->_mat4_col) ? vm->_mat4_col->count : 0; }
size_t mf_vm_get_count_bool(mf_vm* vm) { return (vm && vm->_bool_col) ? vm->_bool_col->count : 0; }
