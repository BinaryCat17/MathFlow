#include "mf_compiler_internal.h"
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
    {"Input", MF_NODE_INPUT},
    
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
    
    // --- Matrix ---
    {"MatMul", MF_NODE_MATMUL},
    {"Transpose", MF_NODE_TRANSPOSE},
    {"Inverse", MF_NODE_INVERSE},
    
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
    {"CumSum", MF_NODE_CUMSUM},
    {"Filter", MF_NODE_COMPRESS},

    // --- State ---
    {"Memory", MF_NODE_MEMORY},

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

// --- String Map Utils ---

static u32 fnv1a_hash(const char* str) {
    u32 hash = 2166136261u;
    while (*str) {
        hash ^= (u8)*str++;
        hash *= 16777619u;
    }
    return hash;
}

void mf_map_init(mf_str_map* map, size_t capacity, mf_arena* arena) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = MF_ARENA_PUSH(arena, mf_map_entry, capacity);
    memset(map->entries, 0, sizeof(mf_map_entry) * capacity);
}

void mf_map_put(mf_str_map* map, const char* key, u32 value) {
    u32 hash = fnv1a_hash(key);
    size_t idx = hash % map->capacity;
    while (map->entries[idx].key != NULL) {
        if (strcmp(map->entries[idx].key, key) == 0) {
            map->entries[idx].value = value; 
            return;
        }
        idx = (idx + 1) % map->capacity;
    }
    map->entries[idx].key = key;
    map->entries[idx].value = value;
    map->count++;
}

bool mf_map_get(mf_str_map* map, const char* key, u32* out_val) {
    u32 hash = fnv1a_hash(key);
    size_t idx = hash % map->capacity;
    while (map->entries[idx].key != NULL) {
        if (strcmp(map->entries[idx].key, key) == 0) {
            *out_val = map->entries[idx].value;
            return true;
        }
        idx = (idx + 1) % map->capacity;
    }
    return false;
}

char* arena_strdup(mf_arena* arena, const char* str) {
    size_t len = strlen(str);
    char* copy = MF_ARENA_PUSH(arena, char, len + 1);
    strcpy(copy, str);
    return copy;
}

// --- Tensor Parsing ---

void parse_constant_tensor(cJSON* val, mf_tensor* t, mf_arena* arena) {
    if (cJSON_IsNumber(val)) {
        // Scalar F32
        t->dtype = MF_DTYPE_F32;
        t->ndim = 0;
        t->size = 1;
        t->capacity_bytes = sizeof(f32);
        t->data = MF_ARENA_PUSH(arena, f32, 1);
        *((f32*)t->data) = (f32)val->valuedouble;
    } 
    else if (cJSON_IsBool(val)) {
        // Scalar Bool (U8)
        t->dtype = MF_DTYPE_U8;
        t->ndim = 0;
        t->size = 1;
        t->capacity_bytes = sizeof(u8);
        t->data = MF_ARENA_PUSH(arena, u8, 1);
        *((u8*)t->data) = (u8)(cJSON_IsTrue(val) ? 1 : 0);
    }
    else if (cJSON_IsString(val)) {
        // String -> I32 Hash
        t->dtype = MF_DTYPE_I32;
        t->ndim = 0;
        t->size = 1;
        t->capacity_bytes = sizeof(int32_t);
        t->data = MF_ARENA_PUSH(arena, int32_t, 1);
        *((int32_t*)t->data) = (int32_t)fnv1a_hash(val->valuestring);
    }
    else if (cJSON_IsArray(val)) {
        int count = cJSON_GetArraySize(val);
        if (count == 0) return; // Empty tensor? 
        
        // Check nesting and type
        cJSON* first = cJSON_GetArrayItem(val, 0);
        
        if (cJSON_IsNumber(first)) {
            // Rank 1 (Vector F32)
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
             // Rank 1 (Vector String -> I32)
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
                if (cJSON_IsString(item))
                    data[i++] = (int32_t)fnv1a_hash(item->valuestring);
                else 
                    data[i++] = 0;
            }
        }
    }
}

// --- Main Parse Function ---

bool mf_compile_load_json(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena) {
    cJSON* root = cJSON_Parse(json_str);
    if (!root) {
        printf("Error parsing JSON.\n");
        return false;
    }

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
        
        // ID
        cJSON* j_id = cJSON_GetObjectItem(node, "id");
        if (cJSON_IsString(j_id)) ir_node->id = arena_strdup(arena, j_id->valuestring);
        else ir_node->id = "unknown";
        
        mf_map_put(&map, ir_node->id, i);

        // Type
        char* type_str = cJSON_GetObjectItem(node, "type")->valuestring;
        ir_node->type = mf_node_type_from_string(type_str);

        // Data (Constant Tensor)
        if (ir_node->type == MF_NODE_INPUT) {
            cJSON* data = cJSON_GetObjectItem(node, "data");
            if (data) {
                cJSON* val = cJSON_GetObjectItem(data, "value");
                if (val) {
                    parse_constant_tensor(val, &ir_node->constant, arena);
                }
            }
        }
        else if (ir_node->type == MF_NODE_MEMORY) {
            // Memory Node: Parse 'init' value
             cJSON* init_val = cJSON_GetObjectItem(node, "init");
             if (init_val) {
                 parse_constant_tensor(init_val, &ir_node->constant, arena);
             } else {
                 // Default to scalar 0
                 ir_node->constant.dtype = MF_DTYPE_F32;
                 ir_node->constant.ndim = 0;
                 ir_node->constant.size = 1;
                 ir_node->constant.data = MF_ARENA_PUSH(arena, f32, 1);
                 *((f32*)ir_node->constant.data) = 0.0f;
             }
        }
        i++;
    }

    // Pass 2: Links
    cJSON* links = cJSON_GetObjectItem(root, "links");
    if (links) {
        int l_count = cJSON_GetArraySize(links);
        out_ir->links = MF_ARENA_PUSH(arena, mf_ir_link, l_count);
        out_ir->link_count = l_count;
        out_ir->link_cap = l_count;

        int j = 0;
        cJSON* link = NULL;
        cJSON_ArrayForEach(link, links) {
            mf_ir_link* ir_link = &out_ir->links[j++];
            
            // Src ID
            cJSON* j_src = cJSON_GetObjectItem(link, "src");
            char key[64];
            if (cJSON_IsString(j_src)) snprintf(key, 64, "%s", j_src->valuestring);
            else key[0] = '\0';
            mf_map_get(&map, key, &ir_link->src_node_idx);

            // Dst ID
            cJSON* j_dst = cJSON_GetObjectItem(link, "dst");
            if (cJSON_IsString(j_dst)) snprintf(key, 64, "%s", j_dst->valuestring);
            else key[0] = '\0';
            mf_map_get(&map, key, &ir_link->dst_node_idx);

            ir_link->src_port = (u32)cJSON_GetObjectItem(link, "src_port")->valueint;
            ir_link->dst_port = (u32)cJSON_GetObjectItem(link, "dst_port")->valueint;
        }
    }

    cJSON_Delete(root);
    return true;
}
