#ifndef MF_PASSES_H
#define MF_PASSES_H

#include <mathflow/base/mf_json.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/base/mf_memory.h>

// --- Pass: AST -> IR (Lowering & Validation) ---
// Converts the AST into the Semantic Graph IR.
// - Resolves Node Types and Enums
// - Validates Data schemas
// - Resolves Port Names to Indices

typedef struct {
    mf_graph_ir* ir;
    mf_arena* arena;
    const char* base_path; // For resolving sub-graphs
} mf_pass_ctx;

bool mf_pass_lower(mf_ast_graph* ast, mf_graph_ir* out_ir, mf_arena* arena, const char* base_path);

// --- Pass: Inline Subgraphs ---
// Recursively expands MF_NODE_CALL into flattened nodes.
// Handles port remapping and unique ID generation.
bool mf_pass_inline(mf_graph_ir* ir, mf_arena* arena);

#endif // MF_PASSES_H
