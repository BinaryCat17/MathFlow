#include "mf_compiler_internal.h"
#include "mf_passes.h"
#include <mathflow/base/mf_json.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <string.h>

bool mf_compile_load_json_ir(const char* json_path, mf_graph_ir* out_ir, mf_arena* arena, mf_compiler_diag* diag) {
    char* json_content = mf_file_read(json_path, arena);
    if (!json_content) {
        mf_source_loc loc = {json_path, 0, 0};
        mf_compiler_diag_report(diag, loc, "Could not read file");
        return false;
    }

    // 1. Parse JSON -> AST (Source Tracking)
    mf_ast_graph* ast = mf_json_parse_graph(json_content, arena);
    if (!ast) {
        mf_source_loc loc = {json_path, 0, 0};
        mf_compiler_diag_report(diag, loc, "Failed to parse JSON AST");
        return false;
    }

    // 2. Lower AST -> IR (Validation & Type resolution)
    if (!mf_pass_lower(ast, out_ir, arena, json_path, diag)) {
        return false;
    }

    return true;
}

// Main Entry Point exposed in mf_compiler.h
bool mf_compile_load_json(const char* json_path, mf_graph_ir* out_ir, mf_arena* arena, mf_compiler_diag* diag) {
    // 1. Load the Root Graph (Raw IR)
    if (!mf_compile_load_json_ir(json_path, out_ir, arena, diag)) {
        return false;
    }

    // 2. Run the Inline Pass (Expand Subgraphs recursively)
    if (!mf_pass_inline(out_ir, arena, diag)) {
        return false;
    }

    return true;
}