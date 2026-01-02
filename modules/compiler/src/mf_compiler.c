#include <mathflow/compiler/mf_compiler.h>
#include "mf_compiler_internal.h"
#include "mf_passes.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

// Note: Logic moved to sub-modules (mf_json_parser.c, mf_semantics.c, mf_codegen.c)

// --- Diagnostics ---

void mf_compiler_diag_init(mf_compiler_diag* diag, mf_arena* arena) {
    memset(diag, 0, sizeof(mf_compiler_diag));
    diag->error_capacity = 32;
    diag->errors = MF_ARENA_PUSH(arena, mf_compiler_error, diag->error_capacity);
}

void mf_compiler_diag_report(mf_compiler_diag* diag, mf_source_loc loc, const char* fmt, ...) {
    if (!diag) return;
    diag->has_error = true;
    if (diag->error_count >= diag->error_capacity) return;

    mf_compiler_error* err = &diag->errors[diag->error_count++];
    err->loc = loc;

    va_list args;
    va_start(args, fmt);
    vsnprintf(err->message, sizeof(err->message), fmt, args);
    va_end(args);

    // Also log to console for immediate feedback during development
    MF_LOG_ERROR("%s:%u:%u: error: %s", loc.file ? loc.file : "unknown", loc.line, loc.column, err->message);
}

// --- Compilation ---

mf_program* mf_compile(mf_graph_ir* ir, const mf_compile_contract* contract, mf_arena* arena, mf_compiler_diag* diag) {
    // 0. Optimizations
    if (!mf_pass_fuse(ir, diag)) {
        return NULL;
    }

    // 1. Sort
    size_t sorted_count = 0;
    mf_ir_node** sorted = mf_topo_sort(ir, arena, &sorted_count);
    if (!sorted) {
        mf_source_loc loc = {0};
        mf_compiler_diag_report(diag, loc, "Cycle detected in graph or sorting failed.");
        return NULL; 
    }

    // 2. Static Analysis (Types & Shapes)
    if (!mf_pass_analyze(ir, sorted, sorted_count, contract, diag)) {
        return NULL;
    }

    // 2.5 Strict Architectural Validation
    if (!mf_pass_validate(ir, sorted, sorted_count, diag)) {
        return NULL;
    }

    // 2a. Register Allocation (Liveness Analysis)
    if (!mf_pass_liveness(ir, sorted, sorted_count, diag)) {
        return NULL;
    }

    // 2b. Domain Splitting (Multi-Domain Support)
    if (!mf_pass_domain_split(ir, diag)) {
        return NULL;
    }

    // 3. Allocate Program Structure
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta.magic = MF_BINARY_MAGIC;
    prog->meta.version = MF_BINARY_VERSION;

    // 4. Emit Code (Tensors, Instructions, State)
    if (!mf_codegen_emit(prog, ir, sorted, sorted_count, arena)) {
        mf_source_loc loc = {0};
        mf_compiler_diag_report(diag, loc, "Code generation failed.");
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

    // 4. Tasks
    if (prog->meta.task_count > 0) {
        fwrite(prog->tasks, sizeof(mf_task), prog->meta.task_count, f);
    }

    // 5. Tensor Metadata
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        mf_type_info* info = &prog->tensor_infos[i];
        mf_bin_tensor_desc desc = {0};
        desc.dtype = (u8)info->dtype;
        desc.ndim = info->ndim;
        desc.builtin_id = prog->builtin_ids[i];
        desc.builtin_axis = prog->builtin_axes[i];
        void* data_ptr = prog->tensor_data[i];
        desc.is_constant = (data_ptr != NULL);
        if (info->ndim > 0) {
            memcpy(desc.shape, info->shape, sizeof(i32) * info->ndim);
        }
        
        if (desc.is_constant) {
            desc.data_size = mf_shape_calc_bytes(info->dtype, info->shape, info->ndim);
        }
        
        fwrite(&desc, sizeof(mf_bin_tensor_desc), 1, f);
    }

    // 5. Tensor Data Blob
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        void* data_ptr = prog->tensor_data[i];
        if (data_ptr) {
            mf_type_info* info = &prog->tensor_infos[i];
            size_t sz = mf_shape_calc_bytes(info->dtype, info->shape, info->ndim);
            fwrite(data_ptr, 1, sz, f);
        }
    }

    fclose(f);
    return true;
}
