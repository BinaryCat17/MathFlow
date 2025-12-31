#include "mf_compiler_internal.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>
#include <string.h>
#include <stdio.h>

// Helper to trace back the shape dependency
static mf_ir_node* find_shape_source(mf_graph_ir* ir, mf_ir_node* node) {
    // Limit depth to prevent infinite loops/stack overflow (though graph is DAG)
    for (int i = 0; i < 32; ++i) {
        if (node->type == MF_NODE_INPUT) return node;
        if (node->type == MF_NODE_CONST) return NULL;
        
        // Stop at shape-changing ops (simplification for Phase 27)
        // Ideally we should support partial propagation, but this is Step 3 start.
        if (node->type == MF_NODE_RESHAPE || 
            node->type == MF_NODE_SLICE || 
            node->type == MF_NODE_MATMUL ||
            node->type == MF_NODE_INDEX ||
            node->type == MF_NODE_COMPRESS ||
            node->type == MF_NODE_JOIN) {
            return NULL;
        }

        // Trace up via src1 (primary operand convention)
        mf_ir_node* parent = find_input_source(ir, (u32)(node - ir->nodes), 0);
        if (!parent) return NULL;
        node = parent;
    }
    return NULL;
}

bool mf_codegen_emit(mf_program* prog, mf_graph_ir* ir, mf_ir_node** sorted, size_t sorted_count, mf_arena* arena) {
    prog->meta.reserved_state = 0;
    prog->meta.tensor_count = (u32)ir->node_count; 
    
    // 2. Count symbols (named nodes)
    u32 symbol_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].id && strcmp(ir->nodes[i].id, "unknown") != 0) {
            symbol_count++;
        }
    }
    prog->meta.symbol_count = symbol_count;
    prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, symbol_count);

    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, prog->meta.tensor_count);
    memset(prog->tensors, 0, sizeof(mf_tensor) * prog->meta.tensor_count);

    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, ir->node_count * 3);
    
    // 4. Generation Loop
    size_t instr_count = 0;
    u32 current_symbol = 0;

    for (size_t i = 0; i < sorted_count; ++i) {
        mf_ir_node* node = sorted[i];
        u32 node_idx = (u32)(node - ir->nodes); 
        
        // Assign Register (Tensor Index)
        node->out_reg_idx = (u16)node_idx; 
        
        // Add to Symbol Table
        if (node->id && strcmp(node->id, "unknown") != 0) {
            mf_bin_symbol* sym = &prog->symbols[current_symbol++];
            strncpy(sym->name, node->id, MF_MAX_SYMBOL_NAME - 1);
            sym->name[MF_MAX_SYMBOL_NAME - 1] = '\0';
            sym->name_hash = mf_fnv1a_hash(sym->name); // Compute Hash
            sym->register_idx = node_idx;
            sym->related_name_hash = 0;
            
            sym->flags = 0;
            if (node->type == MF_NODE_INPUT) {
                sym->flags |= MF_SYMBOL_FLAG_INPUT;
            } else if (node->type == MF_NODE_OUTPUT) {
                sym->flags |= MF_SYMBOL_FLAG_OUTPUT;
                
                // Try to find the input that drives this output's shape
                mf_ir_node* source = find_shape_source(ir, node);
                if (source && source->id) {
                    sym->related_name_hash = mf_fnv1a_hash(source->id);
                    MF_LOG_TRACE("Linked Output '%s' to Input '%s' (Shape Propagation)", node->id, source->id);
                }
            } else {
                // Constants and Named Logic nodes are internal state, 
                // neither Input nor Output for the Pipeline interface.
                sym->flags = 0; 
            }
        }

        // Setup Tensor Descriptor
        mf_tensor* t_desc = &prog->tensors[node_idx];
        
        mf_ir_node* s1 = find_input_source(ir, node_idx, 0);
        mf_ir_node* s2 = find_input_source(ir, node_idx, 1);
        mf_ir_node* s3 = find_input_source(ir, node_idx, 2); 

        // 1. CONST
        if (node->type == MF_NODE_CONST) {
            if (!mf_tensor_is_valid(&node->constant)) {
                MF_LOG_ERROR("Const node '%s' has no data.", node->id ? node->id : "unknown");
                return false;
            }
            *t_desc = node->constant; 
            node->out_shape = node->constant;
        } 
        // 2. INPUT
        else if (node->type == MF_NODE_INPUT) {
            if (s1) {
                // Inlined Input node with a connection! Treat as COPY.
                node->out_shape = s1->out_shape;
                *t_desc = node->out_shape;
                t_desc->buffer = NULL;
                t_desc->byte_offset = 0;
            } else {
                // True global input
                *t_desc = node->constant;
                // Note: Keep buffer if valid to allow Engine to initialize global resources from JSON values
                node->out_shape = node->constant;
            }
        }
        // 3. OUTPUT
        else if (node->type == MF_NODE_OUTPUT) {
             if (s1) {
                 node->out_shape = s1->out_shape;
                 *t_desc = node->out_shape;
                 t_desc->buffer = NULL; 
                 t_desc->byte_offset = 0;
             } else {
                 MF_LOG_ERROR("Output node '%s' is not connected.", node->id);
                 return false;
             }
        }
        else {
            // Logic Node (Shape already inferred by mf_pass_analyze)
            *t_desc = node->out_shape;
            t_desc->buffer = NULL; 
            t_desc->byte_offset = 0;
        }

        mf_instruction* inst = &instrs[instr_count];
        inst->dest_idx = node->out_reg_idx;
        if (s1) inst->src1_idx = s1->out_reg_idx;
        if (s2) inst->src2_idx = s2->out_reg_idx;

        switch (node->type) {
            case MF_NODE_CONST:
                break;
            
            case MF_NODE_INPUT: 
                if (s1) {
                    inst->opcode = MF_OP_COPY;
                    inst->src1_idx = s1->out_reg_idx;
                    instr_count++;
                }
                break;
            
            case MF_NODE_OUTPUT:
                inst->opcode = MF_OP_COPY;
                inst->dest_idx = node->out_reg_idx;
                inst->src1_idx = s1 ? s1->out_reg_idx : 0;
                inst->src2_idx = 0;
                instr_count++;
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
            
            case MF_NODE_RANGE: inst->opcode = MF_OP_RANGE; instr_count++; break;
            case MF_NODE_INDEX: 
                inst->opcode = MF_OP_INDEX; 
                if (!s1) inst->src1_idx = inst->dest_idx; // Read Axis from self-constant
                instr_count++; 
                break;
            case MF_NODE_GATHER:
                inst->opcode = MF_OP_GATHER;
                inst->src1_idx = s1 ? s1->out_reg_idx : 0;
                inst->src2_idx = s2 ? s2->out_reg_idx : 0;
                instr_count++;
                break;
            case MF_NODE_CUMSUM: inst->opcode = MF_OP_CUMSUM; instr_count++; break;
            case MF_NODE_COMPRESS: 
                inst->opcode = MF_OP_COMPRESS; 
                inst->src1_idx = s1 ? s1->out_reg_idx : 0;
                inst->src2_idx = s2 ? s2->out_reg_idx : 0;
                instr_count++; 
                break;

            case MF_NODE_SLICE:
                inst->opcode = MF_OP_SLICE;
                inst->src1_idx = s1 ? s1->out_reg_idx : 0;
                inst->src2_idx = s2 ? s2->out_reg_idx : 0;
                instr_count++;
                break;

            case MF_NODE_RESHAPE:
                inst->opcode = MF_OP_RESHAPE;
                inst->src1_idx = s1 ? s1->out_reg_idx : 0;
                inst->src2_idx = s2 ? s2->out_reg_idx : 0;
                instr_count++;
                break;

            case MF_NODE_STEP: inst->opcode = MF_OP_STEP; instr_count++; break;
            case MF_NODE_DOT: inst->opcode = MF_OP_DOT; instr_count++; break;
            case MF_NODE_LENGTH: inst->opcode = MF_OP_LENGTH; instr_count++; break;
            case MF_NODE_JOIN: inst->opcode = MF_OP_JOIN; instr_count++; break;
            
            case MF_NODE_SMOOTHSTEP:
                inst->opcode = MF_OP_SMOOTHSTEP;
                inst->src1_idx = s1 ? s1->out_reg_idx : 0;
                inst->src2_idx = s2 ? s2->out_reg_idx : 0;
                instr_count++;
                break;

            case MF_NODE_MIX:
                if (s1 && s2 && s3) {
                    inst->opcode = MF_OP_SUB;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s2->out_reg_idx;
                    inst->src2_idx = s1->out_reg_idx;
                    instr_count++;
                    
                    inst = &instrs[instr_count];
                    inst->opcode = MF_OP_MUL;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = node->out_reg_idx;
                    inst->src2_idx = s3->out_reg_idx;
                    instr_count++;

                    inst = &instrs[instr_count];
                    inst->opcode = MF_OP_ADD;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = node->out_reg_idx;
                    inst->src2_idx = s1->out_reg_idx;
                    instr_count++;
                }
                break;

            case MF_NODE_CLAMP:
                if (s1 && s2 && s3) {
                    inst->opcode = MF_OP_CLAMP;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    inst->src3_idx = s3->out_reg_idx;
                    instr_count++;
                }
                break;
            
            case MF_NODE_SELECT: 
                if (s1 && s2 && s3) {
                    inst->opcode = MF_OP_SELECT;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s1->out_reg_idx; // Cond
                    inst->src2_idx = s2->out_reg_idx; // True
                    inst->src3_idx = s3->out_reg_idx; // False
                    instr_count++;
                }
                break;

            default: break;
        }
    }

    prog->code = instrs;
    prog->meta.instruction_count = (u32)instr_count;
    
    return true;
}
