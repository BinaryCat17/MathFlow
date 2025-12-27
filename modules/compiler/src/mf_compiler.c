#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/isa/mf_opcodes.h>
#include <cjson/cJSON.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// --- Node Type Mapping ---

typedef struct {
    const char* name;
    mf_node_type type;
} mf_node_map_entry;

// Maps legacy/specific JSON types to Generic IR Nodes
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

// --- String Map (ID -> Index) ---

typedef struct {
    const char* key;
    u32 value; // Node Index
} mf_map_entry;

typedef struct {
    mf_map_entry* entries;
    size_t capacity;
    size_t count;
} mf_str_map;

static u32 fnv1a_hash(const char* str) {
    u32 hash = 2166136261u;
    while (*str) {
        hash ^= (u8)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static void mf_map_init(mf_str_map* map, size_t capacity, mf_arena* arena) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = MF_ARENA_PUSH(arena, mf_map_entry, capacity);
    memset(map->entries, 0, sizeof(mf_map_entry) * capacity);
}

static void mf_map_put(mf_str_map* map, const char* key, u32 value) {
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

static bool mf_map_get(mf_str_map* map, const char* key, u32* out_val) {
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

// --- Helper: Copy String ---
static char* arena_strdup(mf_arena* arena, const char* str) {
    size_t len = strlen(str);
    char* copy = MF_ARENA_PUSH(arena, char, len + 1);
    strcpy(copy, str);
    return copy;
}

// --- Helper: Parse Tensor from JSON Value ---
// Supports: Number (Scalar), Array (Vector/Matrix), String (Hash), Bool
static void parse_constant_tensor(cJSON* val, mf_tensor* t, mf_arena* arena) {
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
        // TODO: Add Rank 2 (Matrix) parsing here if needed for JSON input
    }
}

// --- Parsing ---

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

static mf_ir_node* find_input_source(mf_graph_ir* ir, u32 dst_node_idx, u32 dst_port) {
    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].dst_node_idx == dst_node_idx && ir->links[i].dst_port == dst_port) {
            return &ir->nodes[ir->links[i].src_node_idx];
        }
    }
    return NULL;
}

// --- Topological Sort Helpers ---
typedef struct {
    mf_ir_node** sorted_nodes; 
    u8* visited;
    size_t count;
    mf_graph_ir* ir;
} sort_ctx;

static void visit_node(sort_ctx* ctx, u32 node_idx) {
    if (ctx->visited[node_idx] == 2) return;
    if (ctx->visited[node_idx] == 1) return; // Cycle
    ctx->visited[node_idx] = 1;

    for (size_t i = 0; i < ctx->ir->link_count; ++i) {
        if (ctx->ir->links[i].dst_node_idx == node_idx) {
            visit_node(ctx, ctx->ir->links[i].src_node_idx);
        }
    }
    ctx->visited[node_idx] = 2;
    ctx->sorted_nodes[ctx->count++] = &ctx->ir->nodes[node_idx];
}

static bool mf_infer_shape(mf_ir_node* node, mf_ir_node* s1, mf_ir_node* s2, mf_ir_node* s3) {
    mf_tensor* out = &node->out_shape;
    memset(out, 0, sizeof(mf_tensor));

    switch (node->type) {
        case MF_NODE_ADD: case MF_NODE_SUB: case MF_NODE_MUL: case MF_NODE_DIV:
        case MF_NODE_MIN: case MF_NODE_MAX: case MF_NODE_ATAN2: case MF_NODE_POW:
        case MF_NODE_CLAMP:
        {
            if (s1 && s2) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                
                // Validation: Both must be same shape or one must be scalar
                bool a_scal = (a->size == 1);
                bool b_scal = (b->size == 1);
                
                if (!a_scal && !b_scal && !mf_tensor_same_shape(a, b)) {
                    printf("Error: Shape mismatch in node '%s'. Input shapes: ", node->id);
                    printf("[%d] vs [%d]\n", a->shape[0], b->shape[0]);
                    return false;
                }

                *out = a_scal ? *b : *a;
            }
        } break;

        case MF_NODE_SIN: case MF_NODE_COS: case MF_NODE_ABS: case MF_NODE_SQRT:
        case MF_NODE_FLOOR: case MF_NODE_CEIL: case MF_NODE_NOT:
        {
            // Unary: Copy shape from first input
            if (s1) *out = s1->out_shape;
        } break;

        case MF_NODE_LESS: case MF_NODE_GREATER: case MF_NODE_EQUAL:
        case MF_NODE_AND: case MF_NODE_OR:
        {
            // Comparison/Logic: Shape from larger input, but DType is U8
            if (s1 && s2) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                *out = (a->size >= b->size) ? *a : *b;
                out->dtype = MF_DTYPE_U8;
            }
        } break;

        case MF_NODE_MATMUL:
            if (s1 && s2) {
                // A: [M, K], B: [K, N] -> C: [M, N]
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                if (a->ndim == 2 && b->ndim == 2) {
                    if (a->shape[1] != b->shape[0]) {
                        printf("Error: MatMul shape mismatch in node '%s'. Inner dimensions [%d] and [%d] do not match.\n", node->id, a->shape[1], b->shape[0]);
                        return false;
                    }
                    out->dtype = a->dtype;
                    out->ndim = 2;
                    out->shape[0] = a->shape[0];
                    out->shape[1] = b->shape[1];
                    out->size = out->shape[0] * out->shape[1];
                } else {
                    // Fallback
                    if (a->size != b->size) {
                        printf("Error: MatMul shape mismatch in node '%s'. Sizes %zu and %zu do not match.\n", node->id, a->size, b->size);
                        return false;
                    }
                    *out = *a; 
                }
            }
            break;

        case MF_NODE_TRANSPOSE:
            if (s1) {
                *out = s1->out_shape;
                if (out->ndim == 2) {
                    int32_t tmp = out->shape[0];
                    out->shape[0] = out->shape[1];
                    out->shape[1] = tmp;
                }
            }
            break;

        case MF_NODE_SELECT:
            if (s2) {
                mf_tensor* t = &s2->out_shape;
                mf_tensor* f = s3 ? &s3->out_shape : NULL;

                if (f) {
                    bool t_s = (t->size == 1);
                    bool f_s = (f->size == 1);
                    if (!t_s && !f_s && !mf_tensor_same_shape(t, f)) {
                         printf("Error: Select shape mismatch in node '%s'.\n", node->id);
                         return false;
                    }
                    *out = t_s ? *f : *t;
                } else {
                    *out = *t;
                }
            }
            break;
            
        case MF_NODE_INVERSE:
            if (s1) *out = s1->out_shape;
            break;

        default: break;
    }
    return true;
}

// --- Compilation ---

mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena) {
    // 1. Sort
    mf_ir_node** sorted = MF_ARENA_PUSH(arena, mf_ir_node*, ir->node_count);
    u8* visited = MF_ARENA_PUSH(arena, u8, ir->node_count);
    memset(visited, 0, ir->node_count);

    sort_ctx ctx = { .sorted_nodes = sorted, .visited = visited, .count = 0, .ir = ir };
    for (size_t i = 0; i < ir->node_count; ++i) visit_node(&ctx, (u32)i);

    // 2. Allocate Program
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta.magic = MF_BINARY_MAGIC;
    prog->meta.version = MF_BINARY_VERSION;
    prog->meta.tensor_count = (u32)ir->node_count; // 1 output tensor per node
    
    // Allocate Tensor Table (Descriptors)
    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, prog->meta.tensor_count);
    memset(prog->tensors, 0, sizeof(mf_tensor) * prog->meta.tensor_count);

    // 3. Instruction Generation & Tensor Init
    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, ir->node_count * 2);
    size_t instr_count = 0;

    for (size_t i = 0; i < ctx.count; ++i) {
        mf_ir_node* node = sorted[i];
        u32 node_idx = (u32)(node - ir->nodes); 
        
        // Assign Register (Tensor Index)
        node->out_reg_idx = (u16)node_idx; // Simple 1-to-1 mapping for now
        
        // Setup Tensor Descriptor
        mf_tensor* t_desc = &prog->tensors[node_idx];
        
        mf_ir_node* s1 = find_input_source(ir, node_idx, 0);
        mf_ir_node* s2 = find_input_source(ir, node_idx, 1);
        mf_ir_node* s3 = find_input_source(ir, node_idx, 2); 

        // If INPUT, copy constant data
        if (node->type == MF_NODE_INPUT) {
            *t_desc = node->constant; 
            node->out_shape = node->constant;
        } else {
            // Logic Node
            if (!mf_infer_shape(node, s1, s2, s3)) {
                return NULL; // Validation failed
            }

            // Write predicted shape to program tensor descriptor
            *t_desc = node->out_shape;
            t_desc->data = NULL; 
        }

        mf_instruction* inst = &instrs[instr_count];
        inst->dest_idx = node->out_reg_idx;
        if (s1) inst->src1_idx = s1->out_reg_idx;
        if (s2) inst->src2_idx = s2->out_reg_idx;

        switch (node->type) {
            case MF_NODE_INPUT: 
                // No instruction, just data
                break;
                
            case MF_NODE_ADD: inst->opcode = MF_OP_ADD; instr_count++; break;
            case MF_NODE_SUB: inst->opcode = MF_OP_SUB; instr_count++; break;
            case MF_NODE_MUL: inst->opcode = MF_OP_MUL; instr_count++; break;
            case MF_NODE_DIV: inst->opcode = MF_OP_DIV; instr_count++; break;
            
            case MF_NODE_MIN: inst->opcode = MF_OP_MIN; instr_count++; break;
            case MF_NODE_MAX: inst->opcode = MF_OP_MAX; instr_count++; break;
            case MF_NODE_ABS: inst->opcode = MF_OP_ABS; instr_count++; break;
            case MF_NODE_SQRT: inst->opcode = MF_OP_SQRT; instr_count++; break;
            case MF_NODE_SIN: inst->opcode = MF_OP_SIN; instr_count++; break;
            case MF_NODE_COS: inst->opcode = MF_OP_COS; instr_count++; break;
            
            case MF_NODE_MATMUL: inst->opcode = MF_OP_MATMUL; instr_count++; break;
            case MF_NODE_TRANSPOSE: inst->opcode = MF_OP_TRANSPOSE; instr_count++; break;
            case MF_NODE_INVERSE: inst->opcode = MF_OP_INVERSE; instr_count++; break;
            
            case MF_NODE_FLOOR: inst->opcode = MF_OP_FLOOR; instr_count++; break;
            case MF_NODE_CEIL: inst->opcode = MF_OP_CEIL; instr_count++; break;
            case MF_NODE_ATAN2: inst->opcode = MF_OP_ATAN2; instr_count++; break;
            case MF_NODE_POW: inst->opcode = MF_OP_POW; instr_count++; break;
            
            case MF_NODE_GREATER: inst->opcode = MF_OP_GREATER; instr_count++; break;
            case MF_NODE_LESS: inst->opcode = MF_OP_LESS; instr_count++; break;
            case MF_NODE_EQUAL: inst->opcode = MF_OP_EQUAL; instr_count++; break;
            case MF_NODE_NEQUAL: inst->opcode = MF_OP_NEQUAL; instr_count++; break;
            case MF_NODE_LEQUAL: inst->opcode = MF_OP_LEQUAL; instr_count++; break;
            case MF_NODE_GEQUAL: inst->opcode = MF_OP_GEQUAL; instr_count++; break;
            
            case MF_NODE_AND: inst->opcode = MF_OP_AND; instr_count++; break;
            case MF_NODE_OR: inst->opcode = MF_OP_OR; instr_count++; break;
            case MF_NODE_NOT: inst->opcode = MF_OP_NOT; instr_count++; break;

            case MF_NODE_CLAMP:
                if (s1 && s2 && s3) {
                    // Clamp(Val, Min, Max)
                    // 1. Dest = MAX(Val, Min)
                    inst->opcode = MF_OP_MAX;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                    
                    // 2. Dest = MIN(Dest, Max)
                    inst = &instrs[instr_count];
                    inst->dest_idx = node->out_reg_idx;
                    inst->opcode = MF_OP_MIN;
                    inst->src1_idx = node->out_reg_idx; // Reuse result
                    inst->src2_idx = s3->out_reg_idx;
                    instr_count++;
                }
                break;
            
            case MF_NODE_SELECT: 
                if (s1 && s2) {
                    // Node inputs: 0=Cond, 1=True, 2=False (Optional)
                    
                    inst->opcode = MF_OP_WHERE_TRUE;
                    inst->src1_idx = s1->out_reg_idx; // Cond
                    inst->src2_idx = s2->out_reg_idx; // TrueVal
                    instr_count++;
                    
                    if (s3) {
                        inst = &instrs[instr_count];
                        inst->opcode = MF_OP_WHERE_FALSE;
                        inst->dest_idx = node->out_reg_idx;
                        inst->src1_idx = s1->out_reg_idx; // Cond
                        inst->src2_idx = s3->out_reg_idx; // FalseVal
                        instr_count++;
                    }
                }
                break;

            default: break;
        }
    }

    prog->code = instrs;
    prog->meta.instruction_count = (u32)instr_count;
    
    return prog;
}

bool mf_compile_save_program(const mf_program* prog, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    // 1. Header
    fwrite(&prog->meta, sizeof(mf_bin_header), 1, f);
    
    // 2. Code
    fwrite(prog->code, sizeof(mf_instruction), prog->meta.instruction_count, f);

    // 3. Tensor Metadata
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        mf_bin_tensor_desc desc = {0};
        desc.dtype = (u8)t->dtype;
        desc.ndim = t->ndim;
        desc.is_constant = (t->data != NULL);
        if (t->ndim > 0) memcpy(desc.shape, t->shape, sizeof(i32) * t->ndim);
        
        if (desc.is_constant) desc.data_size = mf_dtype_size(t->dtype) * t->size;
        
        fwrite(&desc, sizeof(mf_bin_tensor_desc), 1, f);
    }

    // 4. Tensor Data Blob
    for (u32 i = 0; i < prog->meta.tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        if (t->data) {
            size_t sz = mf_dtype_size(t->dtype) * t->size;
            fwrite(t->data, 1, sz, f);
        }
    }

    fclose(f);
    return true;
}