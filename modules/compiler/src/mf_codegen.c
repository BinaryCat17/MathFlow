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
            
            MF_LOG_TRACE("CodeGen: Found symbol '%s' at reg %d", sym->name, node_idx);
            
            sym->flags = 0;
            if (node->type == MF_NODE_INPUT) {
                sym->flags |= MF_SYMBOL_FLAG_INPUT;
            } else if (node->type == MF_NODE_OUTPUT) {
                sym->flags |= MF_SYMBOL_FLAG_OUTPUT;
                // ...
            } else {
                sym->flags = 0; 
            }
            MF_LOG_TRACE("CodeGen: Symbol '%s' flags: 0x%02X (Type: %d)", sym->name, sym->flags, node->type);
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
        if (s3) inst->src3_idx = s3->out_reg_idx;

        switch (node->type) {
#define MF_OP(suffix, name, op_val, cat, mask, out_rule, shape_rule, p1, p2, p3) \
            case MF_NODE_##suffix: \
                if (cat == MF_OP_CAT_SPECIAL) { \
                    if (MF_NODE_##suffix == MF_NODE_INPUT && s1) { \
                         inst->opcode = MF_OP_COPY; inst->src1_idx = s1->out_reg_idx; instr_count++; \
                    } else if (MF_NODE_##suffix == MF_NODE_OUTPUT) { \
                         inst->opcode = MF_OP_COPY; inst->src1_idx = s1 ? s1->out_reg_idx : 0; instr_count++; \
                    } else if (MF_NODE_##suffix == MF_NODE_COPY) { \
                         inst->opcode = MF_OP_COPY; inst->src1_idx = s1 ? s1->out_reg_idx : 0; instr_count++; \
                    } \
                } else { \
                    inst->opcode = op_val; \
                    if (MF_NODE_##suffix == MF_NODE_INDEX && !s1) inst->src1_idx = inst->dest_idx; \
                    instr_count++; \
                } break;
            MF_OP_LIST
#undef MF_OP

            default: break;
        }
    }

    prog->code = instrs;
    prog->meta.instruction_count = (u32)instr_count;
    
    return true;
}
