#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/isa/mf_opcodes.h>
#include "mf_compiler_internal.h"
#include <string.h>
#include <stdio.h>

// Note: Logic moved to sub-modules (mf_json_parser.c, mf_semantics.c, mf_codegen.c)

// --- Compilation ---

mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena) {
    // 1. Sort
    size_t sorted_count = 0;
    mf_ir_node** sorted = mf_topo_sort(ir, arena, &sorted_count);

    // 2. Allocate Program Structure
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta.magic = MF_BINARY_MAGIC;
    prog->meta.version = MF_BINARY_VERSION;

    // 3. Emit Code (Tensors, Instructions, State)
    if (!mf_codegen_emit(prog, ir, sorted, sorted_count, arena)) {
        return NULL;
    }
    
    return prog;
}

bool mf_compile_save_program(const mf_program* prog, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    // 1. Header
    fwrite(&prog->meta, sizeof(mf_bin_header), 1, f);
    
    // 2. Code
    fwrite(prog->code, sizeof(mf_instruction), prog->meta.instruction_count, f);

    // 3. Symbol Table
    if (prog->meta.symbol_count > 0) {
        fwrite(prog->symbols, sizeof(mf_bin_symbol), prog->meta.symbol_count, f);
    }

    // 3.5. State Table
    if (prog->meta.state_count > 0) {
        fwrite(prog->state_table, sizeof(mf_bin_state_link), prog->meta.state_count, f);
    }

    // 4. Tensor Metadata
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        mf_bin_tensor_desc desc = {0};
        desc.dtype = (u8)t->dtype;
        desc.ndim = t->ndim;
        desc.is_constant = (t->data != NULL);
        if (t->ndim > 0) memcpy(desc.shape, t->shape, sizeof(i32) * t->ndim);
        
        if (desc.is_constant) desc.data_size = mf_dtype_size(t->dtype) * t->size;
        
        fwrite(&desc, sizeof(mf_bin_tensor_desc), 1, f);
    }

    // 5. Tensor Data Blob
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        if (t->data) {
            size_t sz = mf_dtype_size(t->dtype) * t->size;
            fwrite(t->data, 1, sz, f);
        }
    }

    fclose(f);
    return true;
}