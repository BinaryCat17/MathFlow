#include "mf_compiler_internal.h"
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// --- Node Type Mapping ---

typedef struct {
    const char* name;
    mf_node_type type;
} mf_node_map_entry;

static const mf_node_map_entry NODE_MAP[] = {
    // --- Core ---
    {"Const", MF_NODE_CONST},
    {"Input", MF_NODE_INPUT},
    {"Output", MF_NODE_OUTPUT},
    
    // --- Arithmetic ---
    {"Add", MF_NODE_ADD},
    {"Sub", MF_NODE_SUB},
    {"Mul", MF_NODE_MUL},
    {"Div", MF_NODE_DIV},
    
    // --- Math ---
    {"Min", MF_NODE_MIN},
    {"Max", MF_NODE_MAX},
    {"Clamp", MF_NODE_CLAMP},
    {"Floor", MF_NODE_FLOOR},
    {"Ceil", MF_NODE_CEIL},
    {"Sin", MF_NODE_SIN},
    {"Cos", MF_NODE_COS},
    {"Atan2", MF_NODE_ATAN2},
    {"Mix", MF_NODE_MIX},
    {"Step", MF_NODE_STEP},
    {"SmoothStep", MF_NODE_SMOOTHSTEP},
    
    // --- Matrix ---
    {"MatMul", MF_NODE_MATMUL},
    {"Transpose", MF_NODE_TRANSPOSE},
    {"Inverse", MF_NODE_INVERSE},
    {"Dot", MF_NODE_DOT},
    {"Length", MF_NODE_LENGTH},
    {"Join", MF_NODE_JOIN}, 
    
    // --- Comparison ---
    {"Greater", MF_NODE_GREATER},
    {"Less", MF_NODE_LESS},
    {"Equal", MF_NODE_EQUAL},
    
    // --- Logic ---
    {"And", MF_NODE_AND},
    {"Or", MF_NODE_OR},
    {"Not", MF_NODE_NOT},
    
    // --- Selection ---
    {"Select", MF_NODE_SELECT},

    // --- Array Ops ---
    {"Range", MF_NODE_RANGE},
    {"Index", MF_NODE_INDEX},
    {"Resolution", MF_NODE_RESOLUTION},
    {"CumSum", MF_NODE_CUMSUM},
    {"Filter", MF_NODE_COMPRESS},

    // --- State ---
    {"Memory", MF_NODE_MEMORY},

    // --- SubGraph ---
    {"Call", MF_NODE_CALL},
    {"ExportInput", MF_NODE_EXPORT_INPUT},
    {"ExportOutput", MF_NODE_EXPORT_OUTPUT},

    {NULL, MF_NODE_UNKNOWN}
};

static mf_node_type mf_node_type_from_string(const char* type) {
    for (const mf_node_map_entry* entry = NODE_MAP; entry->name != NULL; ++entry) {
        if (strcmp(type, entry->name) == 0) {
            return entry->type;
        }
    }
    return MF_NODE_UNKNOWN;
}

// --- Tensor Parsing ---

void parse_constant_tensor(cJSON* val, mf_tensor* t, mf_arena* arena) {
    if (cJSON_IsNumber(val)) {
        t->dtype = MF_DTYPE_F32;
        t->ndim = 0;
        t->size = 1;
        t->capacity_bytes = sizeof(f32);
        t->data = MF_ARENA_PUSH(arena, f32, 1);
        *((f32*)t->data) = (f32)val->valuedouble;
    } 
    else if (cJSON_IsBool(val)) {
        t->dtype = MF_DTYPE_U8;
        t->ndim = 0;
        t->size = 1;
        t->capacity_bytes = sizeof(u8);
        t->data = MF_ARENA_PUSH(arena, u8, 1);
        *((u8*)t->data) = (u8)(cJSON_IsTrue(val) ? 1 : 0);
    }
    else if (cJSON_IsString(val)) {
        t->dtype = MF_DTYPE_I32;
        t->ndim = 0;
        t->size = 1;
        t->capacity_bytes = sizeof(int32_t);
        t->data = MF_ARENA_PUSH(arena, int32_t, 1);
        *((int32_t*)t->data) = (int32_t)mf_fnv1a_hash(val->valuestring);
    }
    else if (cJSON_IsArray(val)) {
        int count = cJSON_GetArraySize(val);
        if (count == 0) return;
        cJSON* first = cJSON_GetArrayItem(val, 0);
        if (cJSON_IsNumber(first)) {
            t->dtype = MF_DTYPE_F32;
            t->ndim = 1;
            t->shape[0] = count;
            t->strides[0] = 1;
            t->size = count;
            t->capacity_bytes = count * sizeof(f32);
            f32* data = MF_ARENA_PUSH(arena, f32, count);
            t->data = data;
            int i = 0;
            cJSON* item = NULL;
            cJSON_ArrayForEach(item, val) {
                data[i++] = (f32)item->valuedouble;
            }
        }
        else if (cJSON_IsString(first)) {
            t->dtype = MF_DTYPE_I32;
            t->ndim = 1;
            t->shape[0] = count;
            t->strides[0] = 1;
            t->size = count;
            t->capacity_bytes = count * sizeof(int32_t);
            int32_t* data = MF_ARENA_PUSH(arena, int32_t, count);
            t->data = data;
            int i = 0;
            cJSON* item = NULL;
            cJSON_ArrayForEach(item, val) {
                if (cJSON_IsString(item)) data[i++] = (int32_t)mf_fnv1a_hash(item->valuestring);
                else data[i++] = 0;
            }
        }
    }
}

// --- Parsing (Flat) ---

static bool parse_flat(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena, const char* base_path) {
    cJSON* root = cJSON_Parse(json_str);
    if (!root) return false;

    cJSON* nodes = cJSON_GetObjectItem(root, "nodes");
    if (!nodes) { cJSON_Delete(root); return false; }

    int count = cJSON_GetArraySize(nodes);
    out_ir->nodes = MF_ARENA_PUSH(arena, mf_ir_node, count);
    out_ir->node_count = count;
    out_ir->node_cap = count;

    mf_str_map map;
    mf_map_init(&map, count * 2, arena);

    // Pass 1: Nodes
    int i = 0;
    cJSON* node = NULL;
    cJSON_ArrayForEach(node, nodes) {
        mf_ir_node* ir_node = &out_ir->nodes[i];
        
        cJSON* j_id = cJSON_GetObjectItem(node, "id");
        if (cJSON_IsString(j_id)) ir_node->id = mf_arena_strdup(arena, j_id->valuestring);
        else ir_node->id = "unknown";
        
        mf_map_put(&map, ir_node->id, i);

        char* type_str = cJSON_GetObjectItem(node, "type")->valuestring;
        ir_node->type = mf_node_type_from_string(type_str);

        // Data Parsing
        cJSON* data = cJSON_GetObjectItem(node, "data");
        
        if ((ir_node->type == MF_NODE_INPUT || ir_node->type == MF_NODE_CONST) && data) {
            cJSON* val = cJSON_GetObjectItem(data, "value");
            if (val) parse_constant_tensor(val, &ir_node->constant, arena);
        }
        else if (ir_node->type == MF_NODE_MEMORY) {
             cJSON* init_val = cJSON_GetObjectItem(node, "init");
             if (init_val) parse_constant_tensor(init_val, &ir_node->constant, arena);
             else {
                 ir_node->constant.dtype = MF_DTYPE_F32;
                 ir_node->constant.ndim = 0;
                 ir_node->constant.size = 1;
                 ir_node->constant.data = MF_ARENA_PUSH(arena, f32, 1);
                 *((f32*)ir_node->constant.data) = 0.0f;
             }
        }
        else if (ir_node->type == MF_NODE_CALL && data) {
            cJSON* path_val = cJSON_GetObjectItem(data, "path");
            if (path_val && cJSON_IsString(path_val)) {
                // If base_path is provided, resolve relative paths
                if (base_path) {
                    char* dir = mf_path_get_dir(base_path, arena);
                    ir_node->sub_graph_path = mf_path_join(dir, path_val->valuestring, arena);
                } else {
                    ir_node->sub_graph_path = mf_arena_strdup(arena, path_val->valuestring);
                }
            }
        }
        else if ((ir_node->type == MF_NODE_EXPORT_INPUT || ir_node->type == MF_NODE_EXPORT_OUTPUT) && data) {
            cJSON* idx_val = cJSON_GetObjectItem(data, "index");
            ir_node->constant.dtype = MF_DTYPE_I32;
            ir_node->constant.ndim = 0;
            ir_node->constant.size = 1;
            ir_node->constant.data = MF_ARENA_PUSH(arena, i32, 1);
            if (idx_val) *((i32*)ir_node->constant.data) = (i32)idx_val->valueint;
            else *((i32*)ir_node->constant.data) = 0;
        }

        i++;
    }

    // Pass 2: Links
    cJSON* links = cJSON_GetObjectItem(root, "links");
    if (links) {
        int l_count = cJSON_GetArraySize(links);
        out_ir->links = MF_ARENA_PUSH(arena, mf_ir_link, l_count);
        memset(out_ir->links, 0, sizeof(mf_ir_link) * l_count); // Ensure clean state
        out_ir->link_count = l_count;
        out_ir->link_cap = l_count;

        int j = 0;
        cJSON* link = NULL;
        cJSON_ArrayForEach(link, links) {
            mf_ir_link* ir_link = &out_ir->links[j++];
            
            cJSON* j_src = cJSON_GetObjectItem(link, "src");
            char key[64];
            if (cJSON_IsString(j_src)) snprintf(key, 64, "%s", j_src->valuestring);
            else key[0] = '\0';
            
            if (!mf_map_get(&map, key, &ir_link->src_node_idx)) {
                MF_LOG_ERROR("Link source node '%s' not found.", key);
            }

            cJSON* j_dst = cJSON_GetObjectItem(link, "dst");
            if (cJSON_IsString(j_dst)) snprintf(key, 64, "%s", j_dst->valuestring);
            else key[0] = '\0';
            
            if (!mf_map_get(&map, key, &ir_link->dst_node_idx)) {
                 MF_LOG_ERROR("Link dest node '%s' not found.", key);
            }

            ir_link->src_port = (u32)cJSON_GetObjectItem(link, "src_port")->valueint;
            ir_link->dst_port = (u32)cJSON_GetObjectItem(link, "dst_port")->valueint;
        }
    } else {
        out_ir->link_count = 0;
        out_ir->links = NULL;
    }

    cJSON_Delete(root);
    return true;
}

// --- Expansion Logic ---

static bool has_call_nodes(mf_graph_ir* ir) {
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].type == MF_NODE_CALL) return true;
    }
    return false;
}

static bool expand_graph(mf_graph_ir* src, mf_graph_ir* dst, mf_arena* arena) {
    // 1. Calculate capacity (simplification: assume max 10x expansion per pass, or just count)
    typedef struct LNode { mf_ir_node n; struct LNode* next; } LNode;
    typedef struct LLink { mf_ir_link l; struct LLink* next; } LLink;
    
    LNode* head_node = NULL;
    LNode* tail_node = NULL;
    size_t new_node_count = 0;

    LLink* head_link = NULL;
    LLink* tail_link = NULL;
    size_t new_link_count = 0;

    // Helper to append node
    #define APPEND_NODE(node_val) { \
        LNode* ln = MF_ARENA_PUSH(arena, LNode, 1); \
        ln->n = node_val; ln->next = NULL; \
        if (tail_node) tail_node->next = ln; else head_node = ln; \
        tail_node = ln; \
        new_node_count++; \
    }

    // Helper to append link
    #define APPEND_LINK(link_val) { \
        LLink* ll = MF_ARENA_PUSH(arena, LLink, 1); \
        ll->l = link_val; ll->next = NULL; \
        if (tail_link) tail_link->next = ll; else head_link = ll; \
        tail_link = ll; \
        new_link_count++; \
    }

    mf_str_map global_map;
    mf_map_init(&global_map, 4096, arena); 

    mf_str_map port_map;
    mf_map_init(&port_map, 1024, arena);

    u32 current_idx = 0;

    for (size_t i = 0; i < src->node_count; ++i) {
        mf_ir_node* node = &src->nodes[i];
        
        if (node->type == MF_NODE_CALL) {
            // LOAD SUBGRAPH
            if (!node->sub_graph_path) continue;
            
            char* json_content = mf_file_read(node->sub_graph_path, arena);
            if (!json_content) {
                MF_LOG_ERROR("Could not read subgraph %s", node->sub_graph_path);
                continue;
            }

            mf_graph_ir child_ir;
            // Pass node->sub_graph_path as base_path so nested calls are resolved correctly
            if (!parse_flat(json_content, &child_ir, arena, node->sub_graph_path)) continue;

            // Process Child Nodes
            for (size_t k = 0; k < child_ir.node_count; ++k) {
                mf_ir_node* c_node = &child_ir.nodes[k];
                
                // 1. Generate Prefixed ID
                char* new_id = mf_arena_sprintf(arena, "%s::%s", node->id, c_node->id);
                
                // 2. Register Ports (Exports)
                if (c_node->type == MF_NODE_EXPORT_INPUT) {
                    i32 port_idx = *((i32*)c_node->constant.data);
                    char port_key[128];
                    snprintf(port_key, 128, "%s:i%d", node->id, port_idx); 
                    mf_map_put(&port_map, mf_arena_strdup(arena, port_key), 0); 
                    
                    c_node->id = new_id;
                    mf_map_put(&global_map, new_id, current_idx);
                    
                    mf_map_put(&port_map, mf_arena_strdup(arena, port_key), current_idx); 
                    
                    APPEND_NODE(*c_node);
                    current_idx++;
                }
                else if (c_node->type == MF_NODE_EXPORT_OUTPUT) {
                    i32 port_idx = *((i32*)c_node->constant.data);
                    char port_key[128];
                    snprintf(port_key, 128, "%s:o%d", node->id, port_idx);
                    
                    c_node->id = new_id;
                    mf_map_put(&global_map, new_id, current_idx);
                    mf_map_put(&port_map, mf_arena_strdup(arena, port_key), current_idx);
                    
                    APPEND_NODE(*c_node);
                    current_idx++;
                }
                else {
                    c_node->id = new_id;
                    mf_map_put(&global_map, new_id, current_idx);
                    APPEND_NODE(*c_node);
                    current_idx++;
                }
            }

            // Process Child Links
            for (size_t k = 0; k < child_ir.link_count; ++k) {
                mf_ir_link l = child_ir.links[k];
                const char* src_id = child_ir.nodes[l.src_node_idx].id;
                const char* dst_id = child_ir.nodes[l.dst_node_idx].id;
                
                u32 new_src_idx, new_dst_idx;
                if (mf_map_get(&global_map, src_id, &new_src_idx) && 
                    mf_map_get(&global_map, dst_id, &new_dst_idx)) 
                {
                    l.src_node_idx = new_src_idx;
                    l.dst_node_idx = new_dst_idx;
                    APPEND_LINK(l);
                }
            }
        } 
        else {
            // Normal Node
            mf_map_put(&global_map, node->id, current_idx);
            APPEND_NODE(*node);
            current_idx++;
        }
    }

    // Step 2: Process Parent Links (Rewiring)
    for (size_t i = 0; i < src->link_count; ++i) {
        mf_ir_link l = src->links[i];
        mf_ir_node* src_node = &src->nodes[l.src_node_idx];
        mf_ir_node* dst_node = &src->nodes[l.dst_node_idx];

        u32 final_src_idx = 0;
        u32 final_dst_idx = 0;
        bool drop_link = false;

        // RESOLVE SOURCE
        if (src_node->type == MF_NODE_CALL) {
            char key[128];
            snprintf(key, 128, "%s:o%d", src_node->id, l.src_port);
            if (!mf_map_get(&port_map, key, &final_src_idx)) drop_link = true; 
            else l.src_port = 0; 
        } else {
            mf_map_get(&global_map, src_node->id, &final_src_idx);
        }

        // RESOLVE DEST
        if (dst_node->type == MF_NODE_CALL) {
            char key[128];
            snprintf(key, 128, "%s:i%d", dst_node->id, l.dst_port);
            if (!mf_map_get(&port_map, key, &final_dst_idx)) drop_link = true;
            else l.dst_port = 0;
        } else {
            mf_map_get(&global_map, dst_node->id, &final_dst_idx);
        }

        if (!drop_link) {
            l.src_node_idx = final_src_idx;
            l.dst_node_idx = final_dst_idx;
            APPEND_LINK(l);
        }
    }

    // Step 3: Finalize Output
    dst->node_count = new_node_count;
    dst->nodes = MF_ARENA_PUSH(arena, mf_ir_node, new_node_count);
    size_t ni = 0;
    for (LNode* cur = head_node; cur; cur = cur->next) {
        dst->nodes[ni++] = cur->n;
    }

    dst->link_count = new_link_count;
    dst->links = MF_ARENA_PUSH(arena, mf_ir_link, new_link_count);
    size_t li = 0;
    for (LLink* cur = head_link; cur; cur = cur->next) {
        dst->links[li++] = cur->l;
    }

    return true;
}

// --- Main Entry Point ---

bool mf_compile_load_json(const char* json_path, mf_graph_ir* out_ir, mf_arena* arena) {
    char* json_content = mf_file_read(json_path, arena);
    if (!json_content) {
        MF_LOG_ERROR("Could not read file %s", json_path);
        return false;
    }

    mf_graph_ir initial_ir;
    if (!parse_flat(json_content, &initial_ir, arena, json_path)) return false;

    // Expand Loop
    mf_graph_ir current_ir = initial_ir;
    for (int pass = 0; pass < 10; ++pass) {
        if (!has_call_nodes(&current_ir)) {
            *out_ir = current_ir;
            return true;
        }
        
        mf_graph_ir next_ir;
        if (!expand_graph(&current_ir, &next_ir, arena)) {
            return false;
        }
        current_ir = next_ir;
    }
    
    MF_LOG_ERROR("Max recursion depth reached or expansion failed.");
    return false;
}
