#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/isa/mf_opcodes.h>
#include "mf_compiler_internal.h"
#include "mf_passes.h"
#include <string.h>
#include <stdio.h>

// Note: Logic moved to sub-modules (mf_json_parser.c, mf_semantics.c, mf_codegen.c)

// --- Compilation ---

mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena) {
    // 1. Sort
    size_t sorted_count = 0;
    mf_ir_node** sorted = mf_topo_sort(ir, arena, &sorted_count);
    if (!sorted) return NULL; // Cycle detected

    // 2. Static Analysis (Types & Shapes)
    // This populates node->out_shape and validates compatibility
    if (!mf_pass_analyze(ir, sorted, sorted_count)) {
        return NULL;
    }

    // 3. Allocate Program Structure
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta.magic = MF_BINARY_MAGIC;
    prog->meta.version = MF_BINARY_VERSION;

    // 4. Emit Code (Tensors, Instructions, State)
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

    // 4. Tensor Metadata
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        mf_bin_tensor_desc desc = {0};
        desc.dtype = (u8)t->info.dtype;
        desc.ndim = t->info.ndim;
        void* data_ptr = mf_tensor_data(t);
        desc.is_constant = (data_ptr != NULL);
        if (t->info.ndim > 0) {
            memcpy(desc.shape, t->info.shape, sizeof(i32) * t->info.ndim);
        }
        
        if (desc.is_constant) {
            desc.data_size = mf_tensor_size_bytes(t);
        }
        
        fwrite(&desc, sizeof(mf_bin_tensor_desc), 1, f);
    }

    // 5. Tensor Data Blob
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        void* data_ptr = mf_tensor_data(t);
        if (data_ptr) {
            size_t sz = mf_tensor_size_bytes(t);
            fwrite(data_ptr, 1, sz, f);
        }
    }

    fclose(f);
    return true;
}
