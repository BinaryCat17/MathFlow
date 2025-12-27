#ifndef MF_COMPILER_INTERNAL_H
#define MF_COMPILER_INTERNAL_H

#include <mathflow/compiler/mf_compiler.h>
#include <cjson/cJSON.h>

// --- Internal Helper: String Map ---
typedef struct {
    const char* key;
    u32 value;
} mf_map_entry;

typedef struct {
    mf_map_entry* entries;
    size_t capacity;
    size_t count;
} mf_str_map;

void mf_map_init(mf_str_map* map, size_t capacity, mf_arena* arena);
void mf_map_put(mf_str_map* map, const char* key, u32 value);
bool mf_map_get(mf_str_map* map, const char* key, u32* out_val);
char* arena_strdup(mf_arena* arena, const char* str);

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

#endif // MF_COMPILER_INTERNAL_H
