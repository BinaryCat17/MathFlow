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

bool mf_pass_lower(mf_ast_graph* ast, mf_graph_ir* out_ir, mf_arena* arena, const char* base_path, mf_compiler_diag* diag);

// --- Pass: Inline Subgraphs ---
// Recursively expands MF_NODE_CALL into flattened nodes.
// Handles port remapping and unique ID generation.
bool mf_pass_inline(mf_graph_ir* ir, mf_arena* arena, mf_compiler_diag* diag);

// --- Pass: Static Analysis (Type & Shape Inference) ---
// Runs on the Topologically Sorted graph.
// - Propagates Shapes and DTypes
// - Validates compatibility (Strong Typing)
// - Checks broadcasting rules
bool mf_pass_analyze(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, const mf_compile_contract* contract, mf_compiler_diag* diag);

// --- Pass: Validation (Strict Consistency) ---
// Performs final structural and semantic checks before codegen.
// - Checks Identity compatibility (e.g. SPATIAL into UNIFORM)
// - Checks Domain consistency
bool mf_pass_validate(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, mf_compiler_diag* diag);

// --- Pass: Domain Splitting ---
// Groups nodes into execution tasks based on their output shapes and dependencies.
bool mf_pass_domain_split(mf_graph_ir* ir, mf_compiler_diag* diag);

// --- Pass: Optimization (Instruction Fusion) ---
// Fuses (Mul + Add) into FMA instructions.
bool mf_pass_fuse(mf_graph_ir* ir, mf_compiler_diag* diag);

// --- Pass: Register Allocation (Liveness Analysis) ---
// Minimizes the number of registers by reusing them for non-overlapping lifetimes.
bool mf_pass_liveness(mf_graph_ir* ir, mf_ir_node** sorted, size_t count, mf_compiler_diag* diag);

#endif // MF_PASSES_H
