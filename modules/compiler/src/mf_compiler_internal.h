#ifndef MF_COMPILER_INTERNAL_H
#define MF_COMPILER_INTERNAL_H

#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_json.h>

// Utilities
void mf_ir_parse_window_settings(const mf_json_value* root, mf_graph_ir* out_ir);
mf_ir_node* find_input_source(mf_graph_ir* ir, u32 dst_node_idx, u32 dst_port);
mf_ir_node* mf_ir_find_input_by_name(mf_graph_ir* ir, u32 dst_node_idx, const char* port_name);

mf_ir_node** mf_topo_sort(mf_graph_ir* ir, mf_arena* arena, size_t* out_count);

// --- Internal: CodeGen ---
// Emits instructions into the program
bool mf_codegen_emit(mf_program* prog, mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t sorted_count, mf_arena* arena);

#endif // MF_COMPILER_INTERNAL_H