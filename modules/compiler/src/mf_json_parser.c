#include "mf_compiler_internal.h"
#include "mf_passes.h"
#include <mathflow/base/mf_json.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <string.h>

// This function is used by mf_pass_inline.c for recursive loading
bool mf_compile_load_json_ir(const char* json_path, mf_graph_ir* out_ir, mf_arena* arena) {
    char* json_content = mf_file_read(json_path, arena);
    if (!json_content) {
        MF_LOG_ERROR("Could not read file %s", json_path);
        return false;
    }

    // 1. Parse JSON -> AST (Source Tracking)
    mf_ast_graph* ast = mf_json_parse_graph(json_content, arena);
    if (!ast) {
        MF_LOG_ERROR("Failed to parse JSON AST for %s", json_path);
        return false;
    }

    // 2. Lower AST -> IR (Validation & Type resolution)
    if (!mf_pass_lower(ast, out_ir, arena, json_path)) {
        MF_LOG_ERROR("Lowering pass failed for %s", json_path);
        return false;
    }

    // Note: We do NOT inline here recursively yet. 
    // The top-level caller decides when to inline. 
    // BUT, mf_pass_inline calls THIS function to load subgraphs.
    // So this function should return the "Raw" IR of a single file.
    
    return true;
}

// Main Entry Point exposed in mf_compiler.h
bool mf_compile_load_json(const char* json_path, mf_graph_ir* out_ir, mf_arena* arena) {
    // 1. Load the Root Graph (Raw IR)
    if (!mf_compile_load_json_ir(json_path, out_ir, arena)) {
        return false;
    }

    // 2. Run the Inline Pass (Expand Subgraphs recursively)
    // This will call mf_compile_load_json_ir internally for each Call node found
    if (!mf_pass_inline(out_ir, arena)) {
        return false;
    }

    return true;
}