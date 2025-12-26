#include <mathflow/compiler/mf_compiler.h>
#include <cjson/cJSON.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// --- Node Type Mapping ---

typedef struct {
    const char* name;
    mf_node_type type;
} mf_node_map_entry;

static const mf_node_map_entry NODE_MAP[] = {
    // Inputs
    {"InputFloat", MF_NODE_INPUT_F32},
    {"InputVec2", MF_NODE_INPUT_VEC2},
    {"InputVec3", MF_NODE_INPUT_VEC3},
    {"InputVec4", MF_NODE_INPUT_VEC4},
    {"InputBool", MF_NODE_INPUT_BOOL},
    // Math - F32
    {"AddFloat", MF_NODE_ADD_F32},
    {"SubFloat", MF_NODE_SUB_F32},
    {"MulFloat", MF_NODE_MUL_F32},
    {"DivFloat", MF_NODE_DIV_F32},
    {"MinFloat", MF_NODE_MIN_F32},
    {"MaxFloat", MF_NODE_MAX_F32},
    {"ClampFloat", MF_NODE_CLAMP_F32},
    {"FloorFloat", MF_NODE_FLOOR_F32},
    {"CeilFloat", MF_NODE_CEIL_F32},
    {"SinFloat", MF_NODE_SIN_F32},
    {"CosFloat", MF_NODE_COS_F32},
    {"Atan2Float", MF_NODE_ATAN2_F32},
    
    // Math - Vec3
    {"AddVec3", MF_NODE_ADD_VEC3},
    {"ScaleVec3", MF_NODE_SCALE_VEC3},
    // Compare
    {"GreaterFloat", MF_NODE_GREATER_F32},
    {"LessFloat", MF_NODE_LESS_F32},
    {"EqualFloat", MF_NODE_EQUAL_F32},
    // Logic
    {"And", MF_NODE_AND},
    {"Or", MF_NODE_OR},
    {"Not", MF_NODE_NOT},
    // Select
    {"SelectFloat", MF_NODE_SELECT_F32},
    {"SelectVec3", MF_NODE_SELECT_VEC3},
    {"SelectVec4", MF_NODE_SELECT_VEC4},
    
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

// --- Parsing ---

bool mf_compile_load_json(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena) {
    cJSON* root = cJSON_Parse(json_str);
    if (!root) {
        printf("Error parsing JSON.\n");
        return false;
    }

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
                if (val) {
                    if (ir_node->type == MF_NODE_INPUT_F32) {
                        ir_node->val_f32 = (f32)val->valuedouble;
                    }
                    else if (ir_node->type == MF_NODE_INPUT_BOOL) {
                        ir_node->val_bool = (u8)(cJSON_IsTrue(val) ? 1 : 0);
                    }
                    else if (cJSON_IsArray(val)) {
                        if (ir_node->type == MF_NODE_INPUT_VEC2) {
                            ir_node->val_vec2.x = (f32)cJSON_GetArrayItem(val, 0)->valuedouble;
                            ir_node->val_vec2.y = (f32)cJSON_GetArrayItem(val, 1)->valuedouble;
                        }
                        else if (ir_node->type == MF_NODE_INPUT_VEC3) {
                            ir_node->val_vec3.x = (f32)cJSON_GetArrayItem(val, 0)->valuedouble;
                            ir_node->val_vec3.y = (f32)cJSON_GetArrayItem(val, 1)->valuedouble;
                            ir_node->val_vec3.z = (f32)cJSON_GetArrayItem(val, 2)->valuedouble;
                        }
                        else if (ir_node->type == MF_NODE_INPUT_VEC4) {
                            ir_node->val_vec4.x = (f32)cJSON_GetArrayItem(val, 0)->valuedouble;
                            ir_node->val_vec4.y = (f32)cJSON_GetArrayItem(val, 1)->valuedouble;
                            ir_node->val_vec4.z = (f32)cJSON_GetArrayItem(val, 2)->valuedouble;
                            ir_node->val_vec4.w = (f32)cJSON_GetArrayItem(val, 3)->valuedouble;
                        }
                    }
                }
            }
        }
    }

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

static mf_ir_node* find_node(mf_graph_ir* ir, u32 id) {
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].id == id) return &ir->nodes[i];
    }
    return NULL;
}

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
    mf_ir_node** sorted_nodes; 
    u8* visited;
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

// --- Compilation ---

mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena) {
    // 1. Setup local columns
    mf_column col_f32, col_vec2, col_vec3, col_vec4, col_bool;
    mf_column_init(&col_f32, sizeof(f32), 16, arena);
    mf_column_init(&col_vec2, sizeof(mf_vec2), 16, arena);
    mf_column_init(&col_vec3, sizeof(mf_vec3), 16, arena);
    mf_column_init(&col_vec4, sizeof(mf_vec4), 16, arena);
    mf_column_init(&col_bool, sizeof(u8), 16, arena);

    // 2. Allocation Pass
    u16 f32_head = 0;
    u16 vec2_head = 0;
    u16 vec3_head = 0;
    u16 vec4_head = 0;
    u16 bool_head = 0;
    
    for (size_t i = 0; i < ir->node_count; ++i) {
        mf_ir_node* node = &ir->nodes[i];
        
        switch (node->type) {
            // Inputs
            case MF_NODE_INPUT_F32:
                node->out_reg_idx = f32_head++;
                f32* fptr = (f32*)mf_column_push(&col_f32, NULL, arena); 
                if (fptr) *fptr = node->val_f32;
                break;
            case MF_NODE_INPUT_VEC2:
                node->out_reg_idx = vec2_head++;
                mf_column_push(&col_vec2, &node->val_vec2, arena);
                break;
            case MF_NODE_INPUT_VEC3:
                node->out_reg_idx = vec3_head++;
                mf_column_push(&col_vec3, &node->val_vec3, arena);
                break;
            case MF_NODE_INPUT_VEC4:
                node->out_reg_idx = vec4_head++;
                mf_column_push(&col_vec4, &node->val_vec4, arena);
                break;
            case MF_NODE_INPUT_BOOL:
                node->out_reg_idx = bool_head++;
                mf_column_push(&col_bool, &node->val_bool, arena);
                break;
                
            // Ops
            case MF_NODE_ADD_F32:
            case MF_NODE_SUB_F32:
            case MF_NODE_MUL_F32:
            case MF_NODE_DIV_F32:
            case MF_NODE_MIN_F32:
            case MF_NODE_MAX_F32:
            // CLAMP handled separately
            case MF_NODE_FLOOR_F32:
            case MF_NODE_CEIL_F32:
            case MF_NODE_SIN_F32:
            case MF_NODE_COS_F32:
            case MF_NODE_ATAN2_F32:
            case MF_NODE_SELECT_F32:
                node->out_reg_idx = f32_head++;
                mf_column_push(&col_f32, NULL, arena);
                break;
            
            case MF_NODE_CLAMP_F32:
                f32_head++; // Temp
                node->out_reg_idx = f32_head++; // Result
                mf_column_push(&col_f32, NULL, arena);
                mf_column_push(&col_f32, NULL, arena);
                break;
                
            case MF_NODE_ADD_VEC3:
            case MF_NODE_SCALE_VEC3:
            case MF_NODE_SELECT_VEC3:
                node->out_reg_idx = vec3_head++;
                mf_column_push(&col_vec3, NULL, arena); 
                break;
                
            case MF_NODE_SELECT_VEC4:
                node->out_reg_idx = vec4_head++;
                mf_column_push(&col_vec4, NULL, arena);
                break;
                
            case MF_NODE_GREATER_F32:
            case MF_NODE_LESS_F32:
            case MF_NODE_EQUAL_F32:
            case MF_NODE_AND:
            case MF_NODE_OR:
            case MF_NODE_NOT:
                node->out_reg_idx = bool_head++;
                mf_column_push(&col_bool, NULL, arena);
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
        if (visited[i] == 0) visit_node(&ctx, &ir->nodes[i]);
    }

    // 4. Create Program
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta.magic = MF_BINARY_MAGIC;
    prog->meta.version = MF_BINARY_VERSION;
    prog->meta.f32_count = (u32)col_f32.count;
    prog->meta.vec2_count = (u32)col_vec2.count;
    prog->meta.vec3_count = (u32)col_vec3.count;
    prog->meta.vec4_count = (u32)col_vec4.count;
    prog->meta.mat4_count = 0;
    prog->meta.bool_count = (u32)col_bool.count;
    
    prog->data_f32 = (f32*)col_f32.data;
    prog->data_vec2 = (mf_vec2*)col_vec2.data;
    prog->data_vec3 = (mf_vec3*)col_vec3.data;
    prog->data_vec4 = (mf_vec4*)col_vec4.data;
    prog->data_mat4 = NULL;
    prog->data_bool = (u8*)col_bool.data;

    // 5. Instruction Generation
    size_t instr_capacity = ir->node_count * 2;
    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, instr_capacity);
    size_t instr_count = 0;

    for (size_t i = 0; i < ctx.count; ++i) {
        mf_ir_node* node = sorted[i];
        mf_ir_node* s1 = find_input_source(ir, node->id, 0);
        mf_ir_node* s2 = find_input_source(ir, node->id, 1);
        mf_ir_node* s3 = find_input_source(ir, node->id, 2); 

        mf_instruction* inst = &instrs[instr_count];

        switch (node->type) {
            case MF_NODE_ADD_F32:
            case MF_NODE_SUB_F32:
            case MF_NODE_MUL_F32:
            case MF_NODE_DIV_F32:
            case MF_NODE_MIN_F32:
            case MF_NODE_MAX_F32:
            case MF_NODE_ATAN2_F32:
                if (s1 && s2) {
                    if (node->type == MF_NODE_ADD_F32) inst->opcode = MF_OP_ADD_F32;
                    else if (node->type == MF_NODE_SUB_F32) inst->opcode = MF_OP_SUB_F32;
                    else if (node->type == MF_NODE_MUL_F32) inst->opcode = MF_OP_MUL_F32;
                    else if (node->type == MF_NODE_DIV_F32) inst->opcode = MF_OP_DIV_F32;
                    else if (node->type == MF_NODE_MIN_F32) inst->opcode = MF_OP_MIN_F32;
                    else if (node->type == MF_NODE_MAX_F32) inst->opcode = MF_OP_MAX_F32;
                    else if (node->type == MF_NODE_ATAN2_F32) inst->opcode = MF_OP_ATAN2_F32;
                    
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;

            case MF_NODE_FLOOR_F32:
            case MF_NODE_CEIL_F32:
            case MF_NODE_SIN_F32:
            case MF_NODE_COS_F32:
                if (s1) {
                     if (node->type == MF_NODE_FLOOR_F32) inst->opcode = MF_OP_FLOOR_F32;
                     else if (node->type == MF_NODE_CEIL_F32) inst->opcode = MF_OP_CEIL_F32;
                     else if (node->type == MF_NODE_SIN_F32) inst->opcode = MF_OP_SIN_F32;
                     else if (node->type == MF_NODE_COS_F32) inst->opcode = MF_OP_COS_F32;
                     
                     inst->dest_idx = node->out_reg_idx;
                     inst->src1_idx = s1->out_reg_idx;
                     inst->src2_idx = 0;
                     instr_count++;
                }
                break;

            case MF_NODE_CLAMP_F32:
                if (s1 && s2 && s3) {
                     // 1. Temp = MAX(Val, Min)
                     inst->opcode = MF_OP_MAX_F32;
                     inst->dest_idx = node->out_reg_idx - 1; 
                     inst->src1_idx = s1->out_reg_idx;
                     inst->src2_idx = s2->out_reg_idx;
                     instr_count++;
                     
                     // 2. Res = MIN(Temp, Max)
                     inst = &instrs[instr_count];
                     inst->opcode = MF_OP_MIN_F32;
                     inst->dest_idx = node->out_reg_idx;
                     inst->src1_idx = node->out_reg_idx - 1;
                     inst->src2_idx = s3->out_reg_idx;
                     instr_count++;
                }
                break;
            case MF_NODE_ADD_VEC3:
                if (s1 && s2) {
                    inst->opcode = MF_OP_ADD_VEC3;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
            case MF_NODE_SCALE_VEC3:
                if (s1 && s2) {
                    inst->opcode = MF_OP_SCALE_VEC3;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
            
            // Comparison
            case MF_NODE_GREATER_F32:
                if (s1 && s2) {
                    inst->opcode = MF_OP_GREATER_F32;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
            case MF_NODE_LESS_F32:
                if (s1 && s2) {
                    inst->opcode = MF_OP_LESS_F32;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
            case MF_NODE_EQUAL_F32:
                if (s1 && s2) {
                    inst->opcode = MF_OP_EQUAL_F32;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
                
            // Logic
            case MF_NODE_AND:
                if (s1 && s2) {
                    inst->opcode = MF_OP_AND;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
             case MF_NODE_OR:
                if (s1 && s2) {
                    inst->opcode = MF_OP_OR;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
             case MF_NODE_NOT:
                if (s1) {
                    inst->opcode = MF_OP_NOT;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = 0;
                    instr_count++;
                }
                break;
                
            // Select (Ternary)
            case MF_NODE_SELECT_F32:
                if (s1 && s2 && s3) {
                    instrs[instr_count].opcode = MF_OP_CMOV_FALSE_F32;
                    instrs[instr_count].dest_idx = node->out_reg_idx;
                    instrs[instr_count].src1_idx = s1->out_reg_idx;
                    instrs[instr_count].src2_idx = s3->out_reg_idx;
                    instr_count++;
                    
                    instrs[instr_count].opcode = MF_OP_CMOV_TRUE_F32;
                    instrs[instr_count].dest_idx = node->out_reg_idx;
                    instrs[instr_count].src1_idx = s1->out_reg_idx;
                    instrs[instr_count].src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
                
            case MF_NODE_SELECT_VEC3:
                if (s1 && s2 && s3) {
                    instrs[instr_count].opcode = MF_OP_CMOV_FALSE_VEC3;
                    instrs[instr_count].dest_idx = node->out_reg_idx;
                    instrs[instr_count].src1_idx = s1->out_reg_idx;
                    instrs[instr_count].src2_idx = s3->out_reg_idx;
                    instr_count++;
                    
                    instrs[instr_count].opcode = MF_OP_CMOV_TRUE_VEC3;
                    instrs[instr_count].dest_idx = node->out_reg_idx;
                    instrs[instr_count].src1_idx = s1->out_reg_idx;
                    instrs[instr_count].src2_idx = s2->out_reg_idx;
                    instr_count++;
                }
                break;
                
            case MF_NODE_SELECT_VEC4:
                if (s1 && s2 && s3) {
                    instrs[instr_count].opcode = MF_OP_CMOV_FALSE_VEC4;
                    instrs[instr_count].dest_idx = node->out_reg_idx;
                    instrs[instr_count].src1_idx = s1->out_reg_idx;
                    instrs[instr_count].src2_idx = s3->out_reg_idx;
                    instr_count++;
                    
                    instrs[instr_count].opcode = MF_OP_CMOV_TRUE_VEC4;
                    instrs[instr_count].dest_idx = node->out_reg_idx;
                    instrs[instr_count].src1_idx = s1->out_reg_idx;
                    instrs[instr_count].src2_idx = s2->out_reg_idx;
                    instr_count++;
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
    if (!prog || !path) return false;

    FILE* f = fopen(path, "wb");
    if (!f) return false;

    fwrite(&prog->meta, sizeof(mf_bin_header), 1, f);
    fwrite(prog->code, sizeof(mf_instruction), prog->meta.instruction_count, f);

    // Save Data Blocks
    if (prog->meta.f32_count > 0) fwrite(prog->data_f32, sizeof(f32), prog->meta.f32_count, f);
    if (prog->meta.vec2_count > 0) fwrite(prog->data_vec2, sizeof(mf_vec2), prog->meta.vec2_count, f);
    if (prog->meta.vec3_count > 0) fwrite(prog->data_vec3, sizeof(mf_vec3), prog->meta.vec3_count, f);
    if (prog->meta.vec4_count > 0) fwrite(prog->data_vec4, sizeof(mf_vec4), prog->meta.vec4_count, f);
    if (prog->meta.mat4_count > 0) fwrite(prog->data_mat4, sizeof(mf_mat4), prog->meta.mat4_count, f);
    if (prog->meta.bool_count > 0) fwrite(prog->data_bool, sizeof(u8), prog->meta.bool_count, f);

    fclose(f);
    return true;
}