#include "mf_compiler_internal.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <string.h>
#include <stdio.h>

bool mf_codegen_emit(mf_program* prog, mf_graph_ir* ir, mf_ir_node** sorted, size_t sorted_count, mf_arena* arena) {
    // 0. Find max register index allocated by liveness pass
    u16 max_reg = 0;
    for (size_t i = 0; i < sorted_count; ++i) {
        if (sorted[i]->out_reg_idx > max_reg) max_reg = sorted[i]->out_reg_idx;
    }

    // 0b. Count extra tensors needed for lowering (e.g. SIZE decomposition)
    u32 extra_tensor_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].type == MF_NODE_SIZE) extra_tensor_count++;
    }

    prog->meta.tensor_count = (u32)max_reg + 1 + extra_tensor_count; 
    
    // 1. Count symbols
    u32 symbol_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].id && strcmp(ir->nodes[i].id, "unknown") != 0) symbol_count++;
    }
    prog->meta.symbol_count = symbol_count;
    prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, symbol_count);

    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, prog->meta.tensor_count);
    memset(prog->tensors, 0, sizeof(mf_tensor) * prog->meta.tensor_count);

    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, ir->node_count * 3);
    mf_task* tasks = MF_ARENA_PUSH(arena, mf_task, ir->node_count + 1);
    
    size_t instr_count = 0;
    u32 task_count = 0;
    u32 current_symbol = 0;
    u32 current_domain_node_idx = UINT32_MAX;
    u32 current_extra_idx = 0;

    for (size_t i = 0; i < sorted_count; ++i) {
        mf_ir_node* node = sorted[i];
        u32 node_idx = (u32)(node - ir->nodes); 
        // node->out_reg_idx is already set by liveness pass
        u16 reg_idx = node->out_reg_idx;
        
        // Symbol Table
        if (node->id && strcmp(node->id, "unknown") != 0) {
            mf_bin_symbol* sym = &prog->symbols[current_symbol++];
            strncpy(sym->name, node->id, MF_MAX_SYMBOL_NAME - 1);
            sym->name[MF_MAX_SYMBOL_NAME - 1] = '\0';
            sym->name_hash = mf_fnv1a_hash(sym->name);
            sym->register_idx = reg_idx;
            sym->flags = (node->type == MF_NODE_INPUT) ? MF_SYMBOL_FLAG_INPUT : 
                         (node->type == MF_NODE_OUTPUT) ? MF_SYMBOL_FLAG_OUTPUT : 0;
        }

        // Tensor Descriptor
        mf_tensor* t_desc = &prog->tensors[reg_idx];
        // If multiple nodes share a register, the liveness pass ensures they don't overlap.
        // We only update the descriptor if it was empty, OR we just let the last node win 
        // (which is fine since they should have compatible shapes or we'll resize anyway).
        t_desc->info = node->out_shape.info;
        
        mf_ir_node* s1 = find_input_source(ir, node_idx, 0);
        mf_ir_node* s2 = find_input_source(ir, node_idx, 1);
        mf_ir_node* s3 = find_input_source(ir, node_idx, 2); 

        if (node->type == MF_NODE_CONST) {
            t_desc->buffer = node->constant.buffer;
            t_desc->byte_offset = node->constant.byte_offset;
        }

        // Emit Instruction
        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        uint32_t start_instr_idx = (uint32_t)instr_count;
        bool emitted = false;

        if (node->type == MF_NODE_SIZE && s1) {
            // Lower SIZE to a Constant tensor containing the element count
            t_desc->info.dtype = MF_DTYPE_F32;
            t_desc->info.ndim = 0;
            t_desc->info.shape[0] = 1;
            mf_shape_calc_strides(&t_desc->info);

            size_t count_val = mf_tensor_count(&s1->out_shape);
            f32* count_data = MF_ARENA_PUSH(arena, f32, 1);
            *count_data = (f32)count_val;

            mf_buffer* count_buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            mf_buffer_init_view(count_buf, count_data, sizeof(f32));
            t_desc->buffer = count_buf;
            t_desc->byte_offset = 0;

            // We don't emit any instruction, it's just a constant register now
            emitted = false; 
        } else {
            mf_instruction* inst = &instrs[instr_count];
            inst->dest_idx = reg_idx;
            inst->src1_idx = s1 ? s1->out_reg_idx : 0;
            inst->src2_idx = s2 ? s2->out_reg_idx : 0;
            inst->src3_idx = s3 ? s3->out_reg_idx : 0;

            if (meta->category == MF_OP_CAT_SPECIAL) {
                if (node->type == MF_NODE_INPUT && s1) {
                    inst->opcode = MF_OP_COPY; instr_count++; emitted = true;
                } else if (node->type == MF_NODE_OUTPUT || node->type == MF_NODE_COPY) {
                    inst->opcode = MF_OP_COPY; instr_count++; emitted = true;
                }
            } else {
                inst->opcode = meta->opcode;
                if (node->type == MF_NODE_INDEX && !s1) inst->src1_idx = inst->dest_idx;
                instr_count++; emitted = true;
            }
        }

        if (emitted) {
            // Task splitting based on explicit domain assignment OR Access Pattern
            bool is_global = (meta->access_pattern == MF_ACCESS_GLOBAL);
            bool domain_changed = (current_domain_node_idx == UINT32_MAX || node->domain_node_idx != current_domain_node_idx);

            if (domain_changed || is_global) {
                if (task_count > 0) {
                    tasks[task_count - 1].inst_count = start_instr_idx - tasks[task_count - 1].start_inst;
                }
                tasks[task_count].start_inst = start_instr_idx;
                
                if (is_global && s1) {
                    tasks[task_count].domain_reg = (u32)(s1 - ir->nodes);
                    current_domain_node_idx = UINT32_MAX; 
                } else {
                    tasks[task_count].domain_reg = (node->domain_node_idx == UINT32_MAX) ? node_idx : node->domain_node_idx;
                    current_domain_node_idx = node->domain_node_idx;
                }
                
                task_count++;
            }
        }
    }

    if (task_count > 0) {
        tasks[task_count - 1].inst_count = (u32)instr_count - tasks[task_count - 1].start_inst;
    }

    prog->code = instrs;
    prog->tasks = tasks;
    prog->meta.instruction_count = (u32)instr_count;
    prog->meta.task_count = task_count;
    
    return true;
}