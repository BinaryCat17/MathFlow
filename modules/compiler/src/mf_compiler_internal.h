#ifndef MF_COMPILER_INTERNAL_H
#define MF_COMPILER_INTERNAL_H

#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/base/mf_utils.h>
#include <cjson/cJSON.h>

// --- Internal: Parse ---
// Parses a constant value from JSON into a tensor descriptor + data in arena
void parse_constant_tensor(cJSON* val, mf_tensor* t, mf_arena* arena);

// --- Internal: Semantics ---
// Infers the output shape of a node based on its inputs
bool mf_infer_shape(mf_ir_node* node, mf_ir_node* s1, mf_ir_node* s2, mf_ir_node* s3);

// --- Internal: Graph Utils ---
mf_ir_node* find_input_source(mf_graph_ir* ir, u32 dst_node_idx, u32 dst_port);

// --- Internal: CodeGen ---
// Sorts the graph topologically and returns the sorted array of nodes
// Returns NULL on cycle error (though currently it just skips cycles)
mf_ir_node** mf_topo_sort(mf_graph_ir* ir, mf_arena* arena, size_t* out_count);

// --- Internal: CodeGen ---
// Emits instructions into the program
bool mf_codegen_emit(mf_program* prog, mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t sorted_count, mf_arena* arena);

#endif // MF_COMPILER_INTERNAL_H