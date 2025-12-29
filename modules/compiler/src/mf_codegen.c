#include "mf_compiler_internal.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>
#include <string.h>
#include <stdio.h>

bool mf_codegen_emit(mf_program* prog, mf_graph_ir* ir, mf_ir_node** sorted, size_t sorted_count, mf_arena* arena) {
    // 1. Count State Nodes (Memory)
    u32 state_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].type == MF_NODE_MEMORY) state_count++;
    }
    prog->meta.state_count = state_count;
    prog->meta.tensor_count = (u32)ir->node_count + state_count; 
    
    // 2. Count symbols (named nodes)
    u32 symbol_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].id && strcmp(ir->nodes[i].id, "unknown") != 0) {
            symbol_count++;
        }
    }
    prog->meta.symbol_count = symbol_count;
    prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, symbol_count);

    // 3. Allocate Tables
    if (state_count > 0) {
        prog->state_table = MF_ARENA_PUSH(arena, mf_bin_state_link, state_count);
    }

    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, prog->meta.tensor_count);
    memset(prog->tensors, 0, sizeof(mf_tensor) * prog->meta.tensor_count);

    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, ir->node_count * 2 + state_count);
    
    // Temp array to store write sources for state updates
    u16* write_sources = NULL;
    if (state_count > 0) {
        write_sources = MF_ARENA_PUSH(arena, u16, state_count);
        for(u32 k=0; k<state_count; ++k) write_sources[k] = 0xFFFF;
    }

    // 4. Generation Loop
    size_t instr_count = 0;
    u32 current_symbol = 0;
    u32 current_state = 0;

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
            sym->register_idx = node_idx;
        }

        // Setup Tensor Descriptor
        mf_tensor* t_desc = &prog->tensors[node_idx];
        
        mf_ir_node* s1 = find_input_source(ir, node_idx, 0);
        mf_ir_node* s2 = find_input_source(ir, node_idx, 1);
        mf_ir_node* s3 = find_input_source(ir, node_idx, 2); 

        // 1. CONST
        if (node->type == MF_NODE_CONST) {
            if (node->constant.data == NULL) {
                MF_LOG_ERROR("Const node '%s' has no data.", node->id ? node->id : "unknown");
                return false;
            }
            *t_desc = node->constant; 
            node->out_shape = node->constant;
        } 
        // 2. INPUT
        else if (node->type == MF_NODE_INPUT) {
            *t_desc = node->constant;
            t_desc->data = NULL;
            node->out_shape = node->constant;
        }
        // 3. OUTPUT
        else if (node->type == MF_NODE_OUTPUT) {
             if (s1) {
                 node->out_shape = s1->out_shape;
                 *t_desc = node->out_shape;
                 t_desc->data = NULL; 
             } else {
                 MF_LOG_ERROR("Output node '%s' is not connected.", node->id);
                 return false;
             }
        }
        // 4. MEMORY (State)
        else if (node->type == MF_NODE_MEMORY) {
            // Read Register (Input for this frame)
            *t_desc = node->constant;
            node->out_shape = node->constant;
            
            // Write Register (Output for next frame)
            u32 write_reg = (u32)ir->node_count + current_state;
            mf_tensor* w_desc = &prog->tensors[write_reg];
            *w_desc = node->constant; // Same prototype
            w_desc->data = NULL;      // Variable (Destination)
            
            // Record Link
            mf_bin_state_link* link = &prog->state_table[current_state];
            link->read_reg = node->out_reg_idx;
            link->write_reg = write_reg;
            
            // Save Source for Copy Instruction
            if (s1) {
                // Since Memory nodes break the topo-sort dependency, s1 might not be processed yet.
                // We rely on 1-to-1 mapping: Register Index = Node Index.
                write_sources[current_state] = (u16)(s1 - ir->nodes);
            }
            
            current_state++;
        }
        else {
            // Logic Node
            if (!mf_infer_shape(node, s1, s2, s3)) {
                MF_LOG_ERROR("Shape inference failed for node '%s'.", node->id ? node->id : "unknown");
                return false; 
            }

            *t_desc = node->out_shape;
            t_desc->data = NULL; 
        }

        mf_instruction* inst = &instrs[instr_count];
        inst->dest_idx = node->out_reg_idx;
        if (s1) inst->src1_idx = s1->out_reg_idx;
        if (s2) inst->src2_idx = s2->out_reg_idx;

        switch (node->type) {
            case MF_NODE_CONST:
            case MF_NODE_INPUT: 
                break;
            
            case MF_NODE_OUTPUT:
            case MF_NODE_EXPORT_INPUT:
            case MF_NODE_EXPORT_OUTPUT:
                inst->opcode = MF_OP_COPY;
                inst->dest_idx = node->out_reg_idx;
                inst->src1_idx = s1 ? s1->out_reg_idx : 0;
                inst->src2_idx = 0;
                instr_count++;
                break;
            
            case MF_NODE_MEMORY:
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
            case MF_NODE_INDEX: inst->opcode = MF_OP_INDEX; instr_count++; break;
            case MF_NODE_RESOLUTION: inst->opcode = MF_OP_RESOLUTION; instr_count++; break;
            case MF_NODE_CUMSUM: inst->opcode = MF_OP_CUMSUM; instr_count++; break;
            case MF_NODE_COMPRESS: 
                inst->opcode = MF_OP_COMPRESS; 
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
                    // Dest = Sub(b, a)
                    inst->opcode = MF_OP_SUB;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s2->out_reg_idx;
                    inst->src2_idx = s1->out_reg_idx;
                    instr_count++;
                    
                    // Dest = Mul(Dest, t)
                    inst = &instrs[instr_count];
                    inst->opcode = MF_OP_MUL;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = node->out_reg_idx;
                    inst->src2_idx = s3->out_reg_idx;
                    instr_count++;

                    // Dest = Add(Dest, a)
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
                    // Dest = MAX(Val, Min)
                    inst->opcode = MF_OP_MAX;
                    inst->src1_idx = s1->out_reg_idx;
                    inst->src2_idx = s2->out_reg_idx;
                    instr_count++;
                    
                    // Dest = MIN(Dest, Max)
                    inst = &instrs[instr_count];
                    inst->dest_idx = node->out_reg_idx;
                    inst->opcode = MF_OP_MIN;
                    inst->src1_idx = node->out_reg_idx; 
                    inst->src2_idx = s3->out_reg_idx;
                    instr_count++;
                }
                break;
            
            case MF_NODE_SELECT: 
                if (s1 && s2) {
                    inst->opcode = MF_OP_WHERE_TRUE;
                    inst->src1_idx = s1->out_reg_idx; 
                    inst->src2_idx = s2->out_reg_idx; 
                    instr_count++;
                    
                    if (s3) {
                        inst = &instrs[instr_count];
                        inst->opcode = MF_OP_WHERE_FALSE;
                        inst->dest_idx = node->out_reg_idx;
                        inst->src1_idx = s1->out_reg_idx; 
                        inst->src2_idx = s3->out_reg_idx; 
                        instr_count++;
                    }
                }
                break;

            default: break;
        }
    }

    // 5. Generate State Updates (Write Phase)
    for (u32 s = 0; s < state_count; ++s) {
        if (write_sources[s] != 0xFFFF) {
            mf_bin_state_link* link = &prog->state_table[s];
            mf_instruction* inst = &instrs[instr_count++];
            inst->opcode = MF_OP_COPY;
            inst->dest_idx = (u16)link->write_reg;
            inst->src1_idx = write_sources[s];
            inst->src2_idx = 0;
        }
    }

    prog->code = instrs;
    prog->meta.instruction_count = (u32)instr_count;
    
    return true;
}