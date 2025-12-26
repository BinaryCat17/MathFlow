#include <mathflow/compiler/mf_compiler.h>
#include <cjson/cJSON.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static mf_node_type mf_node_type_from_string(const char* type) {
    if (strcmp(type, "InputFloat") == 0) return MF_NODE_INPUT_F32;
    if (strcmp(type, "InputVec3") == 0) return MF_NODE_INPUT_VEC3;
    if (strcmp(type, "AddVec3") == 0) return MF_NODE_ADD_VEC3;
    if (strcmp(type, "ScaleVec3") == 0) return MF_NODE_SCALE_VEC3;
    return MF_NODE_UNKNOWN;
}

bool mf_compile_load_json(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena) {
    cJSON* root = cJSON_Parse(json_str);
    if (!root) {
        printf("Error parsing JSON.\n");
        return false;
    }

    // 1. Parse Nodes
    cJSON* nodes = cJSON_GetObjectItem(root, "nodes");
    if (nodes) {
        int count = cJSON_GetArraySize(nodes);
        out_ir->nodes = MF_ARENA_PUSH(arena, mf_ir_node, count);
        out_ir->node_count = count;
        out_ir->node_cap = count;

        cJSON* node = NULL;
        int i = 0;
        cJSON_ArrayForEach(node, nodes) {
            mf_ir_node* ir_node = &out_ir->nodes[i++];
            ir_node->id = (u32)cJSON_GetObjectItem(node, "id")->valueint;
            
            char* type_str = cJSON_GetObjectItem(node, "type")->valuestring;
            ir_node->type = mf_node_type_from_string(type_str);

            // Parse Data
            cJSON* data = cJSON_GetObjectItem(node, "data");
            if (data) {
                cJSON* val = cJSON_GetObjectItem(data, "value");
                if (ir_node->type == MF_NODE_INPUT_F32 && val) {
                    ir_node->val_f32 = (f32)val->valuedouble;
                }
                else if (ir_node->type == MF_NODE_INPUT_VEC3 && val && cJSON_IsArray(val)) {
                    ir_node->val_vec3.x = (f32)cJSON_GetArrayItem(val, 0)->valuedouble;
                    ir_node->val_vec3.y = (f32)cJSON_GetArrayItem(val, 1)->valuedouble;
                    ir_node->val_vec3.z = (f32)cJSON_GetArrayItem(val, 2)->valuedouble;
                }
            }
        }
    }

    // 2. Parse Links
    cJSON* links = cJSON_GetObjectItem(root, "links");
    if (links) {
        int count = cJSON_GetArraySize(links);
        out_ir->links = MF_ARENA_PUSH(arena, mf_ir_link, count);
        out_ir->link_count = count;
        out_ir->link_cap = count;

        cJSON* link = NULL;
        int i = 0;
        cJSON_ArrayForEach(link, links) {
            mf_ir_link* ir_link = &out_ir->links[i++];
            ir_link->src_id = (u32)cJSON_GetObjectItem(link, "src")->valueint;
            ir_link->src_port = (u32)cJSON_GetObjectItem(link, "src_port")->valueint;
            ir_link->dst_id = (u32)cJSON_GetObjectItem(link, "dst")->valueint;
            ir_link->dst_port = (u32)cJSON_GetObjectItem(link, "dst_port")->valueint;
        }
    }

    cJSON_Delete(root);
    return true;
}

// Helper: Find node by ID
static mf_ir_node* find_node(mf_graph_ir* ir, u32 id) {
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].id == id) return &ir->nodes[i];
    }
    return NULL;
}

// Helper: Find source node for a specific input port
static mf_ir_node* find_input_source(mf_graph_ir* ir, u32 dst_id, u32 dst_port) {
    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].dst_id == dst_id && ir->links[i].dst_port == dst_port) {
            return find_node(ir, ir->links[i].src_id);
        }
    }
    return NULL;
}

// --- Topological Sort Helpers ---

typedef struct {
    mf_ir_node** sorted_nodes; // Array of pointers to nodes
    u8* visited;               // 0=no, 1=visiting, 2=visited
    size_t count;
    mf_graph_ir* ir;
} sort_ctx;

static void visit_node(sort_ctx* ctx, mf_ir_node* node) {
    size_t idx = node - ctx->ir->nodes;
    
    if (ctx->visited[idx] == 2) return;
    if (ctx->visited[idx] == 1) {
        printf("Cycle detected! Node ID: %d\n", node->id);
        return;
    }
    
    ctx->visited[idx] = 1;

    for (size_t i = 0; i < ctx->ir->link_count; ++i) {
        if (ctx->ir->links[i].dst_id == node->id) {
            mf_ir_node* src = find_node(ctx->ir, ctx->ir->links[i].src_id);
            if (src) visit_node(ctx, src);
        }
    }

    ctx->visited[idx] = 2;
    ctx->sorted_nodes[ctx->count++] = node;
}

mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena) {
    // 1. Setup local columns to accumulate data
    // using vm memory utils (included via header)
    mf_column col_f32, col_vec3;
    mf_column_init(&col_f32, sizeof(f32), 16, arena);
    mf_column_init(&col_vec3, sizeof(mf_vec3), 16, arena);

    // 2. Allocation Pass
    u16 f32_head = 0;
    u16 vec3_head = 0;
    
    for (size_t i = 0; i < ir->node_count; ++i) {
        mf_ir_node* node = &ir->nodes[i];
        
        switch (node->type) {
            case MF_NODE_INPUT_F32:
                node->out_reg_idx = f32_head++;
                f32* fptr = (f32*)mf_column_push(&col_f32, NULL, arena); 
                if (fptr) *fptr = node->val_f32;
                break;
                
            case MF_NODE_INPUT_VEC3:
                node->out_reg_idx = vec3_head++;
                mf_column_push(&col_vec3, &node->val_vec3, arena);
                break;
                
            case MF_NODE_ADD_VEC3:
            case MF_NODE_SCALE_VEC3:
                node->out_reg_idx = vec3_head++;
                mf_column_push(&col_vec3, NULL, arena); 
                break;
                
            default: break;
        }
    }

    // 3. Topological Sort
    mf_ir_node** sorted = MF_ARENA_PUSH(arena, mf_ir_node*, ir->node_count);
    u8* visited = MF_ARENA_PUSH(arena, u8, ir->node_count);
    memset(visited, 0, ir->node_count);

    sort_ctx ctx = { .sorted_nodes = sorted, .visited = visited, .count = 0, .ir = ir };
    
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (visited[i] == 0) {
            visit_node(&ctx, &ir->nodes[i]);
        }
    }

    // 4. Create Program
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta.magic = MF_BINARY_MAGIC;
    prog->meta.version = MF_BINARY_VERSION;
    prog->meta.f32_count = (u32)col_f32.count;
    prog->meta.vec3_count = (u32)col_vec3.count;
    prog->meta.mat4_count = 0;
    
    // Assign data pointers (pointing to data inside arena)
    prog->data_f32 = (f32*)col_f32.data;
    prog->data_vec3 = (mf_vec3*)col_vec3.data;
    prog->data_mat4 = NULL;

    // 5. Instruction Generation Pass
    size_t instr_capacity = ir->node_count;
    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, instr_capacity);
    size_t instr_count = 0;

    for (size_t i = 0; i < ctx.count; ++i) {
        mf_ir_node* node = sorted[i];
        
        if (node->type == MF_NODE_ADD_VEC3) {
            mf_ir_node* src1 = find_input_source(ir, node->id, 0);
            mf_ir_node* src2 = find_input_source(ir, node->id, 1);
            
            if (src1 && src2) {
                mf_instruction* inst = &instrs[instr_count++];
                inst->opcode = MF_OP_ADD_VEC3;
                inst->dest_idx = node->out_reg_idx;
                inst->src1_idx = src1->out_reg_idx;
                inst->src2_idx = src2->out_reg_idx;
            }
        }
        else if (node->type == MF_NODE_SCALE_VEC3) {
            mf_ir_node* src_vec = find_input_source(ir, node->id, 0);
            mf_ir_node* src_scale = find_input_source(ir, node->id, 1);
            
            if (src_vec && src_scale) {
                mf_instruction* inst = &instrs[instr_count++];
                inst->opcode = MF_OP_SCALE_VEC3;
                inst->dest_idx = node->out_reg_idx;
                inst->src1_idx = src_vec->out_reg_idx; // Vec3 col
                inst->src2_idx = src_scale->out_reg_idx; // F32 col
            }
        }
    }
    
    prog->code = instrs;
    prog->meta.instruction_count = (u32)instr_count;
    
    return prog;
}

bool mf_compile_save_program(const mf_program* prog, const char* path) {
    if (!prog || !path) return false;

    FILE* f = fopen(path, "wb");
    if (!f) return false;

    // 1. Header
    fwrite(&prog->meta, sizeof(mf_bin_header), 1, f);

    // 2. Code
    fwrite(prog->code, sizeof(mf_instruction), prog->meta.instruction_count, f);

    // 3. Data
    if (prog->meta.f32_count > 0 && prog->data_f32)
        fwrite(prog->data_f32, sizeof(f32), prog->meta.f32_count, f);
    
    if (prog->meta.vec3_count > 0 && prog->data_vec3)
        fwrite(prog->data_vec3, sizeof(mf_vec3), prog->meta.vec3_count, f);

    if (prog->meta.mat4_count > 0 && prog->data_mat4)
        fwrite(prog->data_mat4, sizeof(mf_mat4), prog->meta.mat4_count, f);

    fclose(f);
    return true;
}