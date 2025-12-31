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
    {"CumSum", MF_NODE_CUMSUM},
    {"Filter", MF_NODE_COMPRESS},
    {"Slice", MF_NODE_SLICE},
    {"Reshape", MF_NODE_RESHAPE},

    // --- SubGraph ---
    {"Call", MF_NODE_CALL},

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

// --- Node Port Registry ---

typedef struct {
    mf_node_type type;
    const char* port_name;
    u32 port_index;
} mf_node_port_entry;

static const mf_node_port_entry PORT_MAP[] = {
    // Binary Ops
    {MF_NODE_ADD, "a", 0}, {MF_NODE_ADD, "b", 1},
    {MF_NODE_SUB, "a", 0}, {MF_NODE_SUB, "b", 1},
    {MF_NODE_MUL, "a", 0}, {MF_NODE_MUL, "b", 1},
    {MF_NODE_DIV, "a", 0}, {MF_NODE_DIV, "b", 1},
    {MF_NODE_MIN, "a", 0}, {MF_NODE_MIN, "b", 1},
    {MF_NODE_MAX, "a", 0}, {MF_NODE_MAX, "b", 1},
    {MF_NODE_POW, "base", 0}, {MF_NODE_POW, "exp", 1},
    {MF_NODE_ATAN2, "y", 0}, {MF_NODE_ATAN2, "x", 1},

    // Unary Ops
    {MF_NODE_SIN, "x", 0}, {MF_NODE_SIN, "in", 0},
    {MF_NODE_COS, "x", 0}, {MF_NODE_COS, "in", 0},
    {MF_NODE_ABS, "x", 0}, {MF_NODE_ABS, "in", 0},
    {MF_NODE_SQRT, "x", 0}, {MF_NODE_SQRT, "in", 0},
    {MF_NODE_FLOOR, "x", 0}, {MF_NODE_FLOOR, "in", 0},
    {MF_NODE_CEIL, "x", 0}, {MF_NODE_CEIL, "in", 0},
    {MF_NODE_NOT, "in", 0},
    {MF_NODE_LENGTH, "x", 0}, {MF_NODE_LENGTH, "in", 0},
    {MF_NODE_TRANSPOSE, "in", 0},
    {MF_NODE_INVERSE, "in", 0},
    {MF_NODE_NORMALIZE, "in", 0},

    // Matrix
    {MF_NODE_MATMUL, "a", 0}, {MF_NODE_MATMUL, "b", 1},
    {MF_NODE_DOT, "a", 0}, {MF_NODE_DOT, "b", 1},
    {MF_NODE_JOIN, "a", 0}, {MF_NODE_JOIN, "b", 1},

    // Selection / Mix
    {MF_NODE_SELECT, "cond", 0}, {MF_NODE_SELECT, "true", 1}, {MF_NODE_SELECT, "false", 2},
    {MF_NODE_MIX, "a", 0}, {MF_NODE_MIX, "b", 1}, {MF_NODE_MIX, "t", 2},
    {MF_NODE_CLAMP, "x", 0}, {MF_NODE_CLAMP, "min", 1}, {MF_NODE_CLAMP, "max", 2},
    {MF_NODE_SMOOTHSTEP, "x", 0}, {MF_NODE_SMOOTHSTEP, "edges", 1},
    {MF_NODE_STEP, "edge", 0}, {MF_NODE_STEP, "x", 1},

    // Comparison
    {MF_NODE_GREATER, "a", 0}, {MF_NODE_GREATER, "b", 1},
    {MF_NODE_LESS, "a", 0}, {MF_NODE_LESS, "b", 1},
    {MF_NODE_EQUAL, "a", 0}, {MF_NODE_EQUAL, "b", 1},
    {MF_NODE_NEQUAL, "a", 0}, {MF_NODE_NEQUAL, "b", 1},
    {MF_NODE_LEQUAL, "a", 0}, {MF_NODE_LEQUAL, "b", 1},
    {MF_NODE_GEQUAL, "a", 0}, {MF_NODE_GEQUAL, "b", 1},

    // Logic
    {MF_NODE_AND, "a", 0}, {MF_NODE_AND, "b", 1},
    {MF_NODE_OR, "a", 0}, {MF_NODE_OR, "b", 1},
    {MF_NODE_XOR, "a", 0}, {MF_NODE_XOR, "b", 1},

    // Array
    {MF_NODE_RANGE, "count", 0},
    {MF_NODE_SLICE, "in", 0}, {MF_NODE_SLICE, "range", 1},
    {MF_NODE_RESHAPE, "in", 0}, {MF_NODE_RESHAPE, "shape", 1},
    {MF_NODE_CUMSUM, "in", 0},
    {MF_NODE_COMPRESS, "in", 0}, {MF_NODE_COMPRESS, "mask", 1},
    {MF_NODE_INDEX, "axis", 0}, 

    {MF_NODE_INPUT, "out", 0},
    {MF_NODE_OUTPUT, "in", 0},
    {MF_NODE_CONST, "out", 0},

    {MF_NODE_UNKNOWN, NULL, 0}
};

static u32 mf_node_get_port_index(mf_node_type type, const char* port_name) {
    for (const mf_node_port_entry* entry = PORT_MAP; entry->port_name != NULL; ++entry) {
        if (entry->type == type && strcmp(entry->port_name, port_name) == 0) {
            return entry->port_index;
        }
    }
    
    // Default to 0, but if we want to be strict, we could return an error code
    return 0;
}

// --- Tensor Parsing ---

static mf_dtype mf_parse_dtype(const char* s) {
    if (!s) return MF_DTYPE_F32;
    if (strcmp(s, "f32") == 0) return MF_DTYPE_F32;
    if (strcmp(s, "i32") == 0) return MF_DTYPE_I32;
    if (strcmp(s, "bool") == 0 || strcmp(s, "u8") == 0) return MF_DTYPE_U8;
    return MF_DTYPE_F32;
}

static void _init_scalar_info(mf_type_info* info, mf_dtype dtype) {
    info->dtype = dtype;
    info->ndim = 0;
}

static void _init_vector_info(mf_type_info* info, mf_dtype dtype, int count) {
    info->dtype = dtype;
    info->ndim = 1;
    info->shape[0] = count;
    info->strides[0] = 1;
}

void parse_constant_tensor(cJSON* val, mf_tensor* t, mf_arena* arena) {
    if (cJSON_IsNumber(val)) {
        _init_scalar_info(&t->info, MF_DTYPE_F32);
        size_t bytes = sizeof(f32);
        mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
        void* mem = MF_ARENA_PUSH(arena, u8, bytes);
        mf_buffer_init_view(buf, mem, bytes);
        t->buffer = buf;
        t->byte_offset = 0;
        *((f32*)mem) = (f32)val->valuedouble;
    } 
    else if (cJSON_IsBool(val)) {
        _init_scalar_info(&t->info, MF_DTYPE_U8);
        size_t bytes = sizeof(u8);
        mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
        void* mem = MF_ARENA_PUSH(arena, u8, bytes);
        mf_buffer_init_view(buf, mem, bytes);
        t->buffer = buf;
        t->byte_offset = 0;
        *((u8*)mem) = (u8)(cJSON_IsTrue(val) ? 1 : 0);
    }
    else if (cJSON_IsString(val)) {
        _init_scalar_info(&t->info, MF_DTYPE_I32);
        size_t bytes = sizeof(int32_t);
        mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
        void* mem = MF_ARENA_PUSH(arena, u8, bytes);
        mf_buffer_init_view(buf, mem, bytes);
        t->buffer = buf;
        t->byte_offset = 0;
        *((int32_t*)mem) = (int32_t)mf_fnv1a_hash(val->valuestring);
    }
    else if (cJSON_IsArray(val)) {
        int count = cJSON_GetArraySize(val);
        if (count == 0) return;
        cJSON* first = cJSON_GetArrayItem(val, 0);
        if (cJSON_IsNumber(first)) {
            _init_vector_info(&t->info, MF_DTYPE_F32, count);
            size_t bytes = count * sizeof(f32);
            mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            mf_buffer_init_view(buf, mem, bytes);
            t->buffer = buf;
            t->byte_offset = 0;
            f32* data = (f32*)mem;
            int i = 0;
            cJSON* item = NULL;
            cJSON_ArrayForEach(item, val) {
                data[i++] = (f32)item->valuedouble;
            }
        }
        else if (cJSON_IsString(first)) {
            _init_vector_info(&t->info, MF_DTYPE_I32, count);
            size_t bytes = count * sizeof(int32_t);
            mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            mf_buffer_init_view(buf, mem, bytes);
            t->buffer = buf;
            t->byte_offset = 0;
            int32_t* data = (int32_t*)mem;
            int i = 0;
            cJSON* item = NULL;
            cJSON_ArrayForEach(item, val) {
                if (cJSON_IsString(item)) data[i++] = (int32_t)mf_fnv1a_hash(item->valuestring);
                else data[i++] = 0;
            }
        }
    }
}

// --- Validation Helpers ---

static bool validate_keys(cJSON* obj, const char** allowed, size_t count, const char* context) {
    cJSON* item = NULL;
    cJSON_ArrayForEach(item, obj) {
        if (!item->string) continue;
        bool found = false;
        for (size_t i = 0; i < count; ++i) {
            if (strcmp(item->string, allowed[i]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            MF_LOG_ERROR("Unknown field '%s' in %s object", item->string, context);
            return false;
        }
    }
    return true;
}

static const char* VALID_ROOT_KEYS[] = {"nodes", "links"};
static const char* VALID_NODE_KEYS[] = {"id", "type", "data"};
static const char* VALID_LINK_KEYS[] = {"src", "dst", "src_port", "dst_port"};

// Per-type data keys
static const char* VALID_DATA_INPUT[] = {"shape", "dtype"};
static const char* VALID_DATA_CONST[] = {"value"};
static const char* VALID_DATA_INDEX[] = {"axis"};
static const char* VALID_DATA_CALL[] = {"path"};

// --- Parsing (Flat) ---

static bool parse_flat(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena, const char* base_path) {
    cJSON* root = cJSON_Parse(json_str);
    if (!root) {
        MF_LOG_ERROR("JSON parse error before: %s", cJSON_GetErrorPtr());
        return false;
    }

    if (!validate_keys(root, VALID_ROOT_KEYS, 2, "Graph")) {
        cJSON_Delete(root);
        return false;
    }

    cJSON* nodes = cJSON_GetObjectItem(root, "nodes");
    if (!nodes) {
        MF_LOG_ERROR("Invalid graph JSON: missing 'nodes' array.");
        cJSON_Delete(root);
        return false;
    }

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
        if (!validate_keys(node, VALID_NODE_KEYS, 3, "Node")) {
            cJSON_Delete(root);
            return false;
        }

        mf_ir_node* ir_node = &out_ir->nodes[i];
        
        cJSON* j_id = cJSON_GetObjectItem(node, "id");
        if (cJSON_IsString(j_id)) ir_node->id = mf_arena_strdup(arena, j_id->valuestring);
        else ir_node->id = "unknown";
        
        mf_map_put(&map, ir_node->id, i);

        char* type_str = cJSON_GetObjectItem(node, "type")->valuestring;
        ir_node->type = mf_node_type_from_string(type_str);

        // Data Parsing
        cJSON* data = cJSON_GetObjectItem(node, "data");
        
        if (data) {
            if (ir_node->type == MF_NODE_INPUT) {
                 if (!validate_keys(data, VALID_DATA_INPUT, 2, "Input Data")) {
                     cJSON_Delete(root);
                     return false;
                 }
                 cJSON* j_shape = cJSON_GetObjectItem(data, "shape");
                 cJSON* j_dtype = cJSON_GetObjectItem(data, "dtype");
                 
                 if (!j_shape) {
                     MF_LOG_ERROR("Input node '%s' missing required 'shape' field.", ir_node->id);
                     cJSON_Delete(root);
                     return false;
                 }

                 if (cJSON_IsArray(j_shape)) {
                     ir_node->constant.info.dtype = j_dtype ? mf_parse_dtype(j_dtype->valuestring) : MF_DTYPE_F32;
                     ir_node->constant.info.ndim = (uint8_t)cJSON_GetArraySize(j_shape);
                     if (ir_node->constant.info.ndim > MF_MAX_DIMS) ir_node->constant.info.ndim = MF_MAX_DIMS;
                     
                     int d_idx = 0;
                     cJSON* dim = NULL;
                     cJSON_ArrayForEach(dim, j_shape) {
                         if (d_idx < MF_MAX_DIMS) {
                             // Allow -1 for dynamic dimension
                             if (dim->valueint < 0) ir_node->constant.info.shape[d_idx] = -1;
                             else ir_node->constant.info.shape[d_idx] = (int32_t)dim->valueint;
                         }
                         d_idx++;
                     }
                     
                     // Recalc strides (careful with -1)
                     int32_t stride = 1;
                     for(int k=ir_node->constant.info.ndim-1; k>=0; --k) {
                         ir_node->constant.info.strides[k] = stride;
                         int32_t dim_size = ir_node->constant.info.shape[k];
                         if (dim_size < 0) dim_size = 0; // Prevent negative stride propagation for now
                         stride *= dim_size;
                     }

                     ir_node->constant.buffer = NULL;
                     ir_node->constant.byte_offset = 0;
                 } else {
                     MF_LOG_ERROR("Input node '%s' 'shape' must be an array.", ir_node->id);
                     cJSON_Delete(root);
                     return false;
                 } 
                 // Note: We deliberately do NOT parse "value" for Inputs anymore. 
                 // Inputs are strictly interface definitions. Use Const for static data.
            }
            else if (ir_node->type == MF_NODE_CONST || ir_node->type == MF_NODE_STEP) {
                if (!validate_keys(data, VALID_DATA_CONST, 1, "Const Data")) {
                    cJSON_Delete(root);
                    return false;
                }
                cJSON* val = cJSON_GetObjectItem(data, "value");
                if (val) parse_constant_tensor(val, &ir_node->constant, arena);
            }
            else if (ir_node->type == MF_NODE_INDEX) {
                if (!validate_keys(data, VALID_DATA_INDEX, 1, "Index Data")) {
                    cJSON_Delete(root);
                    return false;
                }
                cJSON* axis = cJSON_GetObjectItem(data, "axis");
                if (axis && cJSON_IsNumber(axis)) {
                     _init_scalar_info(&ir_node->constant.info, MF_DTYPE_I32);
                     size_t bytes = sizeof(int32_t);
                     mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
                     void* mem = MF_ARENA_PUSH(arena, u8, bytes);
                     mf_buffer_init_view(buf, mem, bytes);
                     ir_node->constant.buffer = buf;
                     ir_node->constant.byte_offset = 0;
                     *((int32_t*)mem) = (int32_t)axis->valueint;
                }
            }
            else if (ir_node->type == MF_NODE_CALL) {
                if (!validate_keys(data, VALID_DATA_CALL, 1, "Call Data")) {
                    cJSON_Delete(root);
                    return false;
                }
                cJSON* path_val = cJSON_GetObjectItem(data, "path");
                if (path_val && cJSON_IsString(path_val)) {
                    if (base_path) {
                        char* dir = mf_path_get_dir(base_path, arena);
                        ir_node->sub_graph_path = mf_path_join(dir, path_val->valuestring, arena);
                    } else {
                        ir_node->sub_graph_path = mf_arena_strdup(arena, path_val->valuestring);
                    }
                }
            }
        }

        i++;
    }

    // Pass 2: Links
    cJSON* links = cJSON_GetObjectItem(root, "links");
    if (links) {
        int l_count = cJSON_GetArraySize(links);
        out_ir->links = MF_ARENA_PUSH(arena, mf_ir_link, l_count);
        memset(out_ir->links, 0, sizeof(mf_ir_link) * l_count); 
        out_ir->link_count = l_count;
        out_ir->link_cap = l_count;

        int j = 0;
        cJSON* link = NULL;
        cJSON_ArrayForEach(link, links) {
            if (!validate_keys(link, VALID_LINK_KEYS, 4, "Link")) {
                cJSON_Delete(root);
                return false;
            }

            mf_ir_link* ir_link = &out_ir->links[j++];
            memset(ir_link, 0, sizeof(mf_ir_link));
            
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

            cJSON* js_port = cJSON_GetObjectItem(link, "src_port");
            if (cJSON_IsString(js_port)) {
                if (out_ir->nodes[ir_link->src_node_idx].type == MF_NODE_CALL) {
                     ir_link->src_port_name = mf_arena_strdup(arena, js_port->valuestring);
                } else {
                     ir_link->src_port = mf_node_get_port_index(out_ir->nodes[ir_link->src_node_idx].type, js_port->valuestring);
                }
            } else {
                MF_LOG_ERROR("Link src_port must be a string (node '%s')", key);
                return false;
            }

            cJSON* jd_port = cJSON_GetObjectItem(link, "dst_port");
            if (cJSON_IsString(jd_port)) {
                if (out_ir->nodes[ir_link->dst_node_idx].type == MF_NODE_CALL) {
                     ir_link->dst_port_name = mf_arena_strdup(arena, jd_port->valuestring);
                } else {
                     ir_link->dst_port = mf_node_get_port_index(out_ir->nodes[ir_link->dst_node_idx].type, jd_port->valuestring);
                }
            } else {
                MF_LOG_ERROR("Link dst_port must be a string (node '%s')", key);
                return false;
            }
        }
    } else {
        out_ir->link_count = 0;
        out_ir->links = NULL;
    }

    cJSON_Delete(root);
    return true;
}

// --- Expansion Logic ---

static bool is_input_connected(mf_graph_ir* ir, u32 node_idx, u32 port_idx) {
    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].dst_node_idx == node_idx && ir->links[i].dst_port == port_idx) {
            return true;
        }
    }
    return false;
}

static bool needs_expansion(mf_graph_ir* ir) {
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].type == MF_NODE_CALL) return true;
        
        if (mf_tensor_is_valid(&ir->nodes[i].constant)) {
            if (ir->nodes[i].type == MF_NODE_INDEX || 
                ir->nodes[i].type == MF_NODE_STEP) 
            {
                if (!is_input_connected(ir, (u32)i, 0)) return true;
            }
        }
    }
    return false;
}

static bool expand_graph(mf_graph_ir* src, mf_graph_ir* dst, mf_arena* arena) {
    typedef struct LNode { mf_ir_node n; struct LNode* next; } LNode;
    typedef struct LLink { mf_ir_link l; struct LLink* next; } LLink;
    
    LNode* head_node = NULL;
    LNode* tail_node = NULL;
    size_t new_node_count = 0;

    LLink* head_link = NULL;
    LLink* tail_link = NULL;
    size_t new_link_count = 0;

    #define APPEND_NODE(node_val) { \
        LNode* ln = MF_ARENA_PUSH(arena, LNode, 1); \
        ln->n = node_val; ln->next = NULL; \
        if (tail_node) tail_node->next = ln; else head_node = ln; \
        tail_node = ln; \
        new_node_count++; \
    }

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
        
        if (mf_tensor_is_valid(&node->constant) && 
           (node->type == MF_NODE_INDEX || node->type == MF_NODE_STEP)) 
        {
            if (!is_input_connected(src, (u32)i, 0)) {
                char* const_id = mf_arena_sprintf(arena, "%s_impl_const", node->id ? node->id : "gen");
                mf_ir_node const_node = {0};
                const_node.id = const_id;
                const_node.type = MF_NODE_CONST;
                const_node.constant = node->constant; 
                
                node->constant.buffer = NULL;
                node->constant.byte_offset = 0;
                
                mf_map_put(&global_map, const_id, current_idx);
                APPEND_NODE(const_node);
                u32 const_idx = current_idx++;
                
                mf_ir_link implicit_link = {0};
                implicit_link.src_node_idx = const_idx;
                implicit_link.dst_node_idx = current_idx; 
                implicit_link.src_port = 0;
                implicit_link.dst_port = 0;
                
                APPEND_LINK(implicit_link);
            }
        }

        if (node->type == MF_NODE_CALL) {
            if (!node->sub_graph_path) continue;
            
            char* json_content = mf_file_read(node->sub_graph_path, arena);
            if (!json_content) {
                MF_LOG_ERROR("Could not read subgraph %s", node->sub_graph_path);
                continue;
            }

            mf_graph_ir child_ir;
            if (!parse_flat(json_content, &child_ir, arena, node->sub_graph_path)) continue;

            const char** child_raw_ids = MF_ARENA_PUSH(arena, const char*, child_ir.node_count);
            for (size_t k = 0; k < child_ir.node_count; ++k) child_raw_ids[k] = child_ir.nodes[k].id;

            for (size_t k = 0; k < child_ir.node_count; ++k) {
                mf_ir_node* c_node = &child_ir.nodes[k];
                const char* raw_id = child_raw_ids[k];
                char* new_id = mf_arena_sprintf(arena, "%s::%s", node->id, raw_id);
                
                if (c_node->type == MF_NODE_INPUT) {
                    char port_key[128];
                    snprintf(port_key, 128, "%s:i:%s", node->id, raw_id); 
                    
                    c_node->id = new_id;
                    mf_map_put(&global_map, new_id, current_idx);
                    mf_map_put(&port_map, mf_arena_strdup(arena, port_key), current_idx); 
                    
                    APPEND_NODE(*c_node);
                    current_idx++;
                }
                else if (c_node->type == MF_NODE_OUTPUT) {
                    // Bypass logic: Find who provides data to this Output node
                    u32 provider_node_idx = 0;
                    bool found = false;
                    for (size_t l = 0; l < child_ir.link_count; ++l) {
                        if (child_ir.links[l].dst_node_idx == (u32)k) {
                            provider_node_idx = child_ir.links[l].src_node_idx;
                            found = true;
                            break;
                        }
                    }

                    if (found) {
                        char port_key[128];
                        snprintf(port_key, 128, "%s:o:%s", node->id, raw_id);
                        // We map the port name to the PROVIDER node ID (prefixed)
                        // IMPORTANT: Use the raw ID of the provider node to avoid double prefixing
                        const char* provider_raw_id = child_raw_ids[provider_node_idx];
                        char* provider_id = mf_arena_sprintf(arena, "%s::%s", node->id, provider_raw_id);
                        
                        mf_map_put_ptr(&port_map, mf_arena_strdup(arena, port_key), provider_id);
                    }
                }
                else {
                    c_node->id = new_id;
                    mf_map_put(&global_map, new_id, current_idx);
                    APPEND_NODE(*c_node);
                    current_idx++;
                }
            }

            for (size_t k = 0; k < child_ir.link_count; ++k) {
                mf_ir_link l = child_ir.links[k];
                if (child_ir.nodes[l.dst_node_idx].type == MF_NODE_OUTPUT) continue; // Skip links to bypassed outputs

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
            mf_map_put(&global_map, node->id, current_idx);
            APPEND_NODE(*node);
            current_idx++;
        }
    }

    for (size_t i = 0; i < src->link_count; ++i) {
        mf_ir_link l = src->links[i];
        mf_ir_node* src_node = &src->nodes[l.src_node_idx];
        mf_ir_node* dst_node = &src->nodes[l.dst_node_idx];

        u32 final_src_idx = 0;
        u32 final_dst_idx = 0;
        bool drop_link = false;

        if (src_node->type == MF_NODE_CALL) {
            char key[128];
            snprintf(key, 128, "%s:o:%s", src_node->id, l.src_port_name ? l.src_port_name : "unknown");
            
            void* resolved_ptr = NULL;
            if (mf_map_get_ptr(&port_map, key, &resolved_ptr)) {
                const char* provider_id = (const char*)resolved_ptr;
                if (!mf_map_get(&global_map, provider_id, &final_src_idx)) drop_link = true;
                l.src_port = 0; 
            } else {
                drop_link = true;
            }
        } else {
            mf_map_get(&global_map, src_node->id, &final_src_idx);
        }

        if (dst_node->type == MF_NODE_CALL) {
            char key[128];
            snprintf(key, 128, "%s:i:%s", dst_node->id, l.dst_port_name ? l.dst_port_name : "unknown");
            if (!mf_map_get(&port_map, key, &final_dst_idx)) drop_link = true;
            else l.dst_port = 0; // Input to a sub-node is port 0 for Input nodes
        } else {
            mf_map_get(&global_map, dst_node->id, &final_dst_idx);
        }

        if (!drop_link) {
            l.src_node_idx = final_src_idx;
            l.dst_node_idx = final_dst_idx;
            APPEND_LINK(l);
        }
    }

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

    mf_graph_ir current_ir = initial_ir;
    for (int pass = 0; pass < 10; ++pass) {
        if (!needs_expansion(&current_ir)) {
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
