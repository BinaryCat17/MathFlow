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

static char* arena_sprintf(mf_arena* arena, const char* fmt, const char* arg1, const char* arg2) {
    // Simple helper for "arg1::arg2"
    size_t len = strlen(arg1) + strlen(arg2) + 2; // +1 for :: (actually 2) + 1 null? 
    // fmt is just for context, we assume "prefix::name"
    char* buf = MF_ARENA_PUSH(arena, char, len + 3); 
    sprintf(buf, "%s::%s", arg1, arg2);
    return buf;
}

// --- IO Helper ---
static char* read_file(const char* path, mf_arena* arena) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = MF_ARENA_PUSH(arena, char, len + 1);
    if (fread(buf, 1, len, f) != len) {
        fclose(f);
        return NULL;
    }
    buf[len] = 0;
    fclose(f);
    return buf;
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
        *((int32_t*)t->data) = (int32_t)fnv1a_hash(val->valuestring);
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
                if (cJSON_IsString(item)) data[i++] = (int32_t)fnv1a_hash(item->valuestring);
                else data[i++] = 0;
            }
        }
    }
}

// --- Parsing (Flat) ---

static bool parse_flat(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena) {
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
        if (cJSON_IsString(j_id)) ir_node->id = arena_strdup(arena, j_id->valuestring);
        else ir_node->id = "unknown";
        
        mf_map_put(&map, ir_node->id, i);

        char* type_str = cJSON_GetObjectItem(node, "type")->valuestring;
        ir_node->type = mf_node_type_from_string(type_str);

        // Data Parsing
        cJSON* data = cJSON_GetObjectItem(node, "data");
        
        if (ir_node->type == MF_NODE_INPUT && data) {
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
                ir_node->sub_graph_path = arena_strdup(arena, path_val->valuestring);
            }
        }
        else if ((ir_node->type == MF_NODE_EXPORT_INPUT || ir_node->type == MF_NODE_EXPORT_OUTPUT) && data) {
            cJSON* idx_val = cJSON_GetObjectItem(data, "index");
            // Store index in constant data (Scalar I32)
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
            mf_map_get(&map, key, &ir_link->src_node_idx);

            cJSON* j_dst = cJSON_GetObjectItem(link, "dst");
            if (cJSON_IsString(j_dst)) snprintf(key, 64, "%s", j_dst->valuestring);
            else key[0] = '\0';
            mf_map_get(&map, key, &ir_link->dst_node_idx);

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
    // We will just push to arena and hope for the best (Arena is large) 
    
    // We need lists to accumulate result
    // Limitation: We can't easily resize arrays in Arena. 
    // We will use a linked list for nodes/links then flatten.
    
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

    // Map for OldIdx -> NewIdx (in this pass)
    // Actually we need to remap IDs? No, we use string IDs for linking in the end?
    // The IR uses indices. We need to reconstruct indices.
    // Strategy:
    // 1. Iterate src nodes.
    // 2. If Normal: Add to list. Keep track of "OldIdx -> NewIdx".
    // 3. If Call: 
    //      Load Child. Prefix Child IDs. Add Child Nodes to list.
    //      Store "CallIdx:Port -> ChildNodeIdx" mappings.
    // 4. Rebuild Links.

    // Better Strategy:
    // IR uses indices. IDs are for debugging/loading.
    // But expanding breaks indices.
    // We need to rely on string IDs during expansion, then re-resolve to indices?
    // Yes. Re-resolution is safest.

    // Step 1: Collect all Nodes (Flattening)
    // To resolve links, we need a Map<StringID, NodeIdx> for the *New* graph.
    mf_str_map global_map;
    mf_map_init(&global_map, 4096, arena); // Assume large enough

    // We also need to handle "CallPort -> ChildNode" mapping for stitching.
    // Key: "CallNodeID:p<PortIndex>" (e.g., "node5:p0") -> "ChildNodeID"
    mf_str_map port_map;
    mf_map_init(&port_map, 1024, arena);

    u32 current_idx = 0;

    for (size_t i = 0; i < src->node_count; ++i) {
        mf_ir_node* node = &src->nodes[i];
        
        if (node->type == MF_NODE_CALL) {
            // LOAD SUBGRAPH
            if (!node->sub_graph_path) continue;
            
            char* json_content = read_file(node->sub_graph_path, arena);
            if (!json_content) {
                printf("Error: Could not read subgraph %s\n", node->sub_graph_path);
                continue;
            }

            mf_graph_ir child_ir;
            if (!parse_flat(json_content, &child_ir, arena)) continue;

            // Process Child Nodes
            for (size_t k = 0; k < child_ir.node_count; ++k) {
                mf_ir_node* c_node = &child_ir.nodes[k];
                
                // 1. Generate Prefixed ID
                char* new_id = arena_sprintf(arena, "%s::%s", node->id, c_node->id); // "call_id::child_id"
                
                // 2. Register Ports (Exports)
                if (c_node->type == MF_NODE_EXPORT_INPUT) {
                    i32 port_idx = *((i32*)c_node->constant.data);
                    char port_key[128];
                    snprintf(port_key, 128, "%s:i%d", node->id, port_idx); // Input to Call -> ExportInput
                    mf_map_put(&port_map, arena_strdup(arena, port_key), 0); // Value unused, we need target ID.
                    
                    // Actually, we need to map "CallNodeID:InputPort" -> "ChildNodeThatUsesExportInput".
                    // Wait, links go TO ExportInput. Then ExportInput goes TO ChildNode.
                    // Flattening removes ExportInput?
                    // Let's keep ExportInput nodes for now, and let optimization remove them?
                    // Or turn them into Identity/Copy?
                    // Let's turn ExportInput/Output into Identity (or just leave as is and let VM handle? No, VM doesn't know).
                    // Best: Treat ExportInput as "Anchor".
                    
                    // Let's just add the node with the new ID.
                    c_node->id = new_id;
                    mf_map_put(&global_map, new_id, current_idx);
                    
                    // Register "Call:i0" -> "Call::ExportInput"
                    mf_map_put(&port_map, arena_strdup(arena, port_key), current_idx); // Map to the new node index
                    
                    APPEND_NODE(*c_node);
                    current_idx++;
                }
                else if (c_node->type == MF_NODE_EXPORT_OUTPUT) {
                    i32 port_idx = *((i32*)c_node->constant.data);
                    char port_key[128];
                    snprintf(port_key, 128, "%s:o%d", node->id, port_idx);
                    
                    c_node->id = new_id;
                    mf_map_put(&global_map, new_id, current_idx);
                    mf_map_put(&port_map, arena_strdup(arena, port_key), current_idx);
                    
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
                // Resolve IDs using the Prefixed IDs
                const char* src_id_old = child_ir.nodes[l.src_node_idx].id; // These are already prefixed? No, child_ir has original pointers?
                // Wait, child_ir.nodes[k].id was modified in place above!
                // So l.src_node_idx points to the node in child_ir array, which has the NEW ID.
                const char* src_id = child_ir.nodes[l.src_node_idx].id;
                const char* dst_id = child_ir.nodes[l.dst_node_idx].id;
                
                // We will resolve indices later. Store IDs?
                // We can't store IDs in mf_ir_link (it has u32).
                // But we know the Global Map has the IDs.
                // So we can look up the New Index immediately!
                
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
            // Comes FROM a Call node (Output)
            // Look up in PortMap: "CallID:o<Port>"
            char key[128];
            snprintf(key, 128, "%s:o%d", src_node->id, l.src_port);
            if (!mf_map_get(&port_map, key, &final_src_idx)) {
                // Port not found?
                drop_link = true; 
            } else {
                // The new source is the ExportOutput node.
                // We need to connect ExportOutput -> Dst.
                // Wait, ExportOutput is an Input to the external world?
                // Inside child: Node -> ExportOutput.
                // Outside: Call -> Node.
                // Result: Node -> ExportOutput -> Node.
                // ExportOutput acts as a relay.
                // We need to ensure ExportOutput behaves like a Copy or Identity.
                l.src_port = 0; // ExportOutput has only 1 input (0) and we treat it as passing through?
                // Actually, ExportOutput isn't an operation. It's a marker.
                // If we treat it as `MF_NODE_INPUT` (Identity) that takes an input?
                // Let's set its type to `MF_NODE_ADD` (0 + x) or just `Copy`?
                // We don't have Copy. `MF_NODE_MAX` with itself?
                // Better: The Compiler's CodeGen should handle ExportOutput by just aliasing the register.
                // For now, let's treat it as a node that exists.
            }
        } else {
            // Normal source
            mf_map_get(&global_map, src_node->id, &final_src_idx);
        }

        // RESOLVE DEST
        if (dst_node->type == MF_NODE_CALL) {
            // Goes TO a Call node (Input)
            // Look up in PortMap: "CallID:i<Port>"
            char key[128];
            snprintf(key, 128, "%s:i%d", dst_node->id, l.dst_port);
            if (!mf_map_get(&port_map, key, &final_dst_idx)) {
                drop_link = true;
            } else {
                l.dst_port = 0; // ExportInput has 1 output (0)
            }
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

bool mf_compile_load_json(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena) {
    mf_graph_ir initial_ir;
    if (!parse_flat(json_str, &initial_ir, arena)) return false;

    // Expand Loop (Max depth 10)
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
    
    printf("Error: Max recursion depth reached or expansion failed.\n");
    return false;
}