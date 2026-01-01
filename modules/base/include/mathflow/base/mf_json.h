#ifndef MF_JSON_H
#define MF_JSON_H

#include <mathflow/base/mf_types.h>
#include <mathflow/base/mf_memory.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    u32 line;
    u32 column;
} mf_json_loc;

typedef enum {
    MF_JSON_VAL_NULL,
    MF_JSON_VAL_BOOL,
    MF_JSON_VAL_NUMBER,
    MF_JSON_VAL_STRING,
    MF_JSON_VAL_ARRAY,
    MF_JSON_VAL_OBJECT
} mf_json_val_type;

typedef struct mf_json_value {
    mf_json_val_type type;
    union {
        bool b;
        double n;
        const char* s;
        struct {
            struct mf_json_value* items;
            size_t count;
        } array;
        struct {
            const char** keys;
            struct mf_json_value* values;
            size_t count;
        } object;
    } as;
    mf_json_loc loc;
} mf_json_value;

// --- Graph Specific AST (Legacy/Helper) ---
typedef struct {
    const char* id;
    const char* type;
    mf_json_value* data;
    mf_json_loc loc;
} mf_ast_node;

typedef struct {
    const char* src;
    const char* dst;
    const char* src_port;
    const char* dst_port;
    mf_json_loc loc;
} mf_ast_link;

typedef struct {
    mf_ast_node* nodes;
    size_t node_count;
    mf_ast_link* links;
    size_t link_count;
    const char** imports;
    size_t import_count;
    const char* source_path;
} mf_ast_graph;

// --- API ---

// Helper to get a value from an object value by key
const mf_json_value* mf_json_get_field(const mf_json_value* obj, const char* key);

// General JSON Parser
mf_json_value* mf_json_parse(const char* json_str, mf_arena* arena);

// Graph Specific Parser (Wraps mf_json_parse and extracts nodes/links)
mf_ast_graph* mf_json_parse_graph(const char* json_str, mf_arena* arena);

#endif // MF_JSON_H