#include "mf_compiler_internal.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

bool mf_codegen_emit(mf_program* prog, mf_graph_ir* ir, mf_ir_node** sorted, size_t sorted_count, mf_arena* arena) {
    u16 max_reg = 0;
    for (size_t i = 0; i < sorted_count; ++i) {
        if (sorted[i]->out_reg_idx > max_reg) max_reg = sorted[i]->out_reg_idx;
    }

    u32 extra_tensor_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].type == MF_NODE_SIZE) extra_tensor_count++;
    }

    prog->meta.tensor_count = (u32)max_reg + 1 + extra_tensor_count; 
    
    u32 symbol_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].id && strcmp(ir->nodes[i].id, "unknown") != 0) symbol_count++;
    }
    prog->meta.symbol_count = symbol_count;
    prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, symbol_count);

    prog->tensor_infos = MF_ARENA_PUSH(arena, mf_type_info, prog->meta.tensor_count);
    memset(prog->tensor_infos, 0, sizeof(mf_type_info) * prog->meta.tensor_count);

    prog->tensor_data = MF_ARENA_PUSH(arena, void*, prog->meta.tensor_count);
    memset(prog->tensor_data, 0, sizeof(void*) * prog->meta.tensor_count);

    prog->builtin_ids = MF_ARENA_PUSH(arena, uint8_t, prog->meta.tensor_count);
    memset(prog->builtin_ids, 0, prog->meta.tensor_count);

    prog->builtin_axes = MF_ARENA_PUSH(arena, uint8_t, prog->meta.tensor_count);
    memset(prog->builtin_axes, 0, prog->meta.tensor_count);

    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, ir->node_count * 3);
    mf_task* tasks = MF_ARENA_PUSH(arena, mf_task, ir->node_count + 1);
    
    size_t instr_count = 0;
    u32 task_count = 0;
    u32 current_symbol = 0;
    u32 current_domain_node_idx = UINT32_MAX;

    for (size_t i = 0; i < sorted_count; ++i) {
        mf_ir_node* node = sorted[i];
        u32 node_idx = (u32)(node - ir->nodes); 
        u16 reg_idx = node->out_reg_idx;
        
        if (node->id && strcmp(node->id, "unknown") != 0) {
            mf_bin_symbol* sym = &prog->symbols[current_symbol++];
            strncpy(sym->name, node->id, MF_MAX_SYMBOL_NAME - 1);
            sym->name[MF_MAX_SYMBOL_NAME - 1] = '\0';
            sym->name_hash = mf_fnv1a_hash(sym->name);
            sym->register_idx = reg_idx;
            sym->builtin_id = node->builtin_id;
            sym->builtin_axis = node->builtin_axis;
            if (node->provider) strncpy(sym->provider, node->provider, MF_MAX_SYMBOL_NAME - 1);
            else sym->provider[0] = '\0';
            sym->flags = (node->type == MF_NODE_INPUT) ? MF_SYMBOL_FLAG_INPUT : 
                         (node->type == MF_NODE_OUTPUT) ? MF_SYMBOL_FLAG_OUTPUT : 0;
        }

        mf_type_info* t_info = &prog->tensor_infos[reg_idx];
        if (node->builtin_id != MF_BUILTIN_NONE) {
            prog->builtin_ids[reg_idx] = (uint8_t)node->builtin_id;
            prog->builtin_axes[reg_idx] = (uint8_t)node->builtin_axis;
        }
        
        size_t current_size = mf_shape_calc_bytes(t_info->dtype, t_info->shape, t_info->ndim);
        size_t new_size = mf_shape_calc_bytes(node->out_info.dtype, node->out_info.shape, node->out_info.ndim);
        if (t_info->ndim == 0 || new_size > current_size) *t_info = node->out_info;

        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        mf_ir_node* inputs[4] = {0};
        for (u8 k = 0; k < 4; ++k) {
            if (meta->ports[k]) inputs[k] = mf_ir_find_input_by_name(ir, node_idx, meta->ports[k]);
        }

        if (node->type == MF_NODE_CONST) prog->tensor_data[reg_idx] = node->const_data;

        uint32_t start_instr_idx = (uint32_t)instr_count;
        bool emitted = false;

        if (node->type == MF_NODE_SIZE && inputs[0]) {
            prog->tensor_infos[reg_idx] = (mf_type_info){MF_DTYPE_F32, 0, {1}, {0}};
            f32* count_data = MF_ARENA_PUSH(arena, f32, 1);
            *count_data = (f32)mf_shape_calc_count(inputs[0]->out_info.shape, inputs[0]->out_info.ndim);
            prog->tensor_data[reg_idx] = count_data;
        } else {
            mf_instruction* inst = &instrs[instr_count];
            memset(inst, 0, sizeof(mf_instruction));
            inst->dest_idx = reg_idx;
            inst->src1_idx = inputs[0] ? inputs[0]->out_reg_idx : 0;
            inst->src2_idx = inputs[1] ? inputs[1]->out_reg_idx : 0;
            inst->src3_idx = inputs[2] ? inputs[2]->out_reg_idx : 0;
            inst->src4_idx = inputs[3] ? inputs[3]->out_reg_idx : 0;
            inst->line = node->loc.line;
            inst->column = node->loc.column;
            memcpy(inst->strides, node->strides, sizeof(i32) * 5);

            if (meta->category == MF_OP_CAT_SPECIAL) {
                if ((node->type == MF_NODE_INPUT && inputs[0]) || node->type == MF_NODE_OUTPUT || node->type == MF_NODE_COPY) {
                    inst->opcode = MF_OP_COPY; instr_count++; emitted = true;
                }
            } else {
                inst->opcode = meta->opcode;
                instr_count++; emitted = true;
            }
        }

        if (emitted) {
            bool is_global = (meta->access_pattern == MF_ACCESS_GLOBAL);
            bool domain_changed = (current_domain_node_idx == UINT32_MAX || node->domain_node_idx != current_domain_node_idx);
            if (domain_changed || is_global) {
                if (task_count > 0) tasks[task_count - 1].inst_count = start_instr_idx - tasks[task_count - 1].start_inst;
                tasks[task_count].start_inst = start_instr_idx;
                if (is_global && inputs[0]) {
                    tasks[task_count].domain_reg = inputs[0]->out_reg_idx;
                    current_domain_node_idx = UINT32_MAX; 
                } else {
                    u32 dom_node_idx = (node->domain_node_idx == UINT32_MAX) ? node_idx : node->domain_node_idx;
                    tasks[task_count].domain_reg = ir->nodes[dom_node_idx].out_reg_idx;
                    current_domain_node_idx = node->domain_node_idx;
                }
                task_count++;
            }
        }
    }

    if (task_count > 0) tasks[task_count - 1].inst_count = (u32)instr_count - tasks[task_count - 1].start_inst;
    prog->code = instrs;
    prog->tasks = tasks;
    prog->meta.instruction_count = (u32)instr_count;
    prog->meta.task_count = task_count;
    return true;
}