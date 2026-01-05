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
    prog->symbols = (symbol_count > 0) ? MF_ARENA_PUSH(arena, mf_bin_symbol, symbol_count) : NULL;

    prog->tensor_infos = MF_ARENA_PUSH(arena, mf_type_info, prog->meta.tensor_count);
    memset(prog->tensor_infos, 0, sizeof(mf_type_info) * prog->meta.tensor_count);

    prog->tensor_data = MF_ARENA_PUSH(arena, void*, prog->meta.tensor_count);
    memset(prog->tensor_data, 0, sizeof(void*) * prog->meta.tensor_count);

    prog->builtin_ids = MF_ARENA_PUSH(arena, uint8_t, prog->meta.tensor_count);
    memset(prog->builtin_ids, 0, prog->meta.tensor_count);

    prog->builtin_axes = MF_ARENA_PUSH(arena, uint8_t, prog->meta.tensor_count);
    memset(prog->builtin_axes, 0, prog->meta.tensor_count);

    prog->tensor_flags = MF_ARENA_PUSH(arena, uint8_t, prog->meta.tensor_count);
    memset(prog->tensor_flags, 0, prog->meta.tensor_count);

    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, ir->node_count * 3);
    mf_task* tasks = MF_ARENA_PUSH(arena, mf_task, ir->node_count * 2);
    
    mf_bin_task_binding* bindings = MF_ARENA_PUSH(arena, mf_bin_task_binding, ir->node_count * 10);
    u32 total_binding_count = 0;

    typedef struct {
        u16 reg_idx;
        i32 byte_stride;
        bool is_reduction;
    } temp_binding;
    temp_binding current_bindings[MF_MAX_REGISTERS];
    u32 current_binding_count = 0;

    size_t instr_count = 0;
    u32 task_count = 0;
    u32 current_symbol = 0;
    u32 current_domain_node_idx = UINT32_MAX;
    u8 current_strategy = MF_STRATEGY_DEFAULT;
    bool needs_sync_scratch = false;

    for (size_t i = 0; i < sorted_count; ++i) {
        mf_ir_node* node = sorted[i];
        u32 node_idx = (u32)(node - ir->nodes); 
        u16 r_idx = node->out_reg_idx;
        
        // --- 1. Symbol Table Entry ---
        if (node->id && strcmp(node->id, "unknown") != 0) {
            mf_bin_symbol* sym = &prog->symbols[current_symbol++];
            strncpy(sym->name, node->id, MF_MAX_SYMBOL_NAME - 1);
            sym->name[MF_MAX_SYMBOL_NAME - 1] = '\0';
            sym->name_hash = mf_fnv1a_hash(sym->name);
            sym->register_idx = r_idx;
            sym->builtin_id = node->builtin_id;
            sym->builtin_axis = node->builtin_axis;
            if (node->provider) strncpy(sym->provider, node->provider, MF_MAX_SYMBOL_NAME - 1);
            else sym->provider[0] = '\0';
            
            sym->flags = (node->type == MF_NODE_INPUT) ? MF_SYMBOL_FLAG_INPUT : 
                         (node->type == MF_NODE_OUTPUT) ? MF_SYMBOL_FLAG_OUTPUT : 0;
            sym->flags |= (node->resource_flags & (MF_RESOURCE_FLAG_READONLY | MF_RESOURCE_FLAG_PERSISTENT | MF_RESOURCE_FLAG_TRANSIENT));

            if (node->type == MF_NODE_INPUT || node->type == MF_NODE_OUTPUT) {
                prog->tensor_flags[r_idx] |= MF_TENSOR_FLAG_ALIAS;
            }
        }

        // --- 2. Register Metadata (with Inflation) ---
        mf_type_info* t_info = &prog->tensor_infos[r_idx];
        size_t node_elements = mf_shape_calc_count(node->out_info.shape, node->out_info.ndim);
        if (node->is_spatial) {
            u32 dom_idx = (node->domain_node_idx == UINT32_MAX) ? node_idx : node->domain_node_idx;
            size_t task_cnt = mf_shape_calc_count(ir->nodes[dom_idx].out_info.shape, ir->nodes[dom_idx].out_info.ndim);
            if (task_cnt > node_elements) node_elements = task_cnt;
        }

        size_t current_elements = mf_shape_calc_count(t_info->shape, t_info->ndim);
        if (t_info->ndim == 0 || node_elements > current_elements) {
            *t_info = node->out_info;
            if (node->is_spatial && node_elements > mf_shape_calc_count(node->out_info.shape, node->out_info.ndim)) {
                t_info->ndim = 1;
                t_info->shape[0] = (int32_t)node_elements;
                mf_shape_calc_strides(t_info);
            }
        }

        if (node->builtin_id != MF_BUILTIN_NONE) {
            prog->builtin_ids[r_idx] = (uint8_t)node->builtin_id;
            prog->builtin_axes[r_idx] = (uint8_t)node->builtin_axis;
            prog->tensor_flags[r_idx] |= MF_TENSOR_FLAG_GENERATOR;
        }

        // --- 3. Instruction Generation ---
        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        mf_ir_node* inputs[4] = {0};
        for (u8 k = 0; k < 4; ++k) if (meta->ports[k]) inputs[k] = mf_ir_find_input_by_name(ir, node_idx, meta->ports[k]);

        if (node->type == MF_NODE_CONST) {
            prog->tensor_data[r_idx] = node->const_data;
            prog->tensor_flags[r_idx] |= MF_TENSOR_FLAG_CONSTANT;
        }

        uint32_t start_instr_idx = (uint32_t)instr_count;
        bool emitted = false;

        if (node->type == MF_NODE_SIZE && inputs[0]) {
            prog->tensor_infos[r_idx] = (mf_type_info){MF_DTYPE_F32, 0, {1}, {0}};
            f32* count_data = MF_ARENA_PUSH(arena, f32, 1);
            *count_data = (f32)mf_shape_calc_count(inputs[0]->out_info.shape, inputs[0]->out_info.ndim);
            prog->tensor_data[r_idx] = count_data;
            prog->tensor_flags[r_idx] |= MF_TENSOR_FLAG_CONSTANT;
        } else if (meta->category != MF_OP_CAT_SPECIAL || node->type == MF_NODE_COPY || node->type == MF_NODE_INPUT || node->type == MF_NODE_OUTPUT) {
            mf_instruction* inst = &instrs[instr_count++];
            memset(inst, 0, sizeof(mf_instruction));
            inst->dest_idx = r_idx;
            inst->src1_idx = inputs[0] ? inputs[0]->out_reg_idx : 0;
            inst->src2_idx = inputs[1] ? inputs[1]->out_reg_idx : 0;
            inst->src3_idx = inputs[2] ? inputs[2]->out_reg_idx : 0;
            inst->src4_idx = inputs[3] ? inputs[3]->out_reg_idx : 0;
            inst->line = (u16)node->loc.line;
            inst->column = (u16)node->loc.column;
            inst->opcode = (meta->category == MF_OP_CAT_SPECIAL) ? MF_OP_COPY : meta->opcode;
            emitted = true;
        }

        // --- 4. Task Management ---
        if (emitted) {
            bool is_sync = (meta->strategy == MF_STRATEGY_TWO_PASS_SYNC);
            bool is_reduction = (meta->strategy == MF_STRATEGY_REDUCTION);
            bool domain_changed = (current_domain_node_idx == UINT32_MAX || node->domain_node_idx != current_domain_node_idx);
            
            if (is_reduction && r_idx < MF_MAX_REGISTERS) prog->tensor_flags[r_idx] |= MF_TENSOR_FLAG_REDUCTION;
            if (is_sync) needs_sync_scratch = true;

            bool needs_split = domain_changed || is_sync || (current_strategy != meta->strategy);

            if (needs_split && task_count > 0) {
                mf_task* prev_task = &tasks[task_count - 1];
                prev_task->inst_count = start_instr_idx - prev_task->start_inst;
                prev_task->binding_offset = total_binding_count;
                prev_task->binding_count = current_binding_count;
                for (u32 b = 0; b < current_binding_count; ++b) {
                    bindings[total_binding_count].reg_idx = current_bindings[b].reg_idx;
                    bindings[total_binding_count].byte_stride = current_bindings[b].byte_stride;
                    bindings[total_binding_count].flags = current_bindings[b].is_reduction ? MF_BINDING_FLAG_REDUCTION : 0;
                    total_binding_count++;
                }
                current_binding_count = 0;
            }

            if (needs_split || task_count == 0) {
                tasks[task_count].start_inst = start_instr_idx;
                tasks[task_count].strategy = meta->strategy;
                u32 dom_node_idx = (node->domain_node_idx == UINT32_MAX) ? node_idx : node->domain_node_idx;
                tasks[task_count].domain_reg = ir->nodes[dom_node_idx].out_reg_idx;
                current_domain_node_idx = node->domain_node_idx;
                current_strategy = meta->strategy;
                task_count++;
            }

            u16 ops[5] = { r_idx, 
                           inputs[0] ? inputs[0]->out_reg_idx : 0,
                           inputs[1] ? inputs[1]->out_reg_idx : 0,
                           inputs[2] ? inputs[2]->out_reg_idx : 0,
                           inputs[3] ? inputs[3]->out_reg_idx : 0 };
            
            for (int k = 0; k < 5; ++k) {
                if (k > 0 && !inputs[k-1]) continue;
                u16 r = ops[k];
                i32 b_stride = node->strides[k] * (i32)mf_dtype_size(prog->tensor_infos[r].dtype);
                bool found = false;
                for (u32 b = 0; b < current_binding_count; ++b) {
                    if (current_bindings[b].reg_idx == r) {
                        if (is_reduction && k == 0) current_bindings[b].is_reduction = true;
                        found = true; break;
                    }
                }
                if (!found) {
                    current_bindings[current_binding_count].reg_idx = r;
                    current_bindings[current_binding_count].byte_stride = b_stride;
                    current_bindings[current_binding_count].is_reduction = (is_reduction && k == 0);
                    current_binding_count++;
                }
            }
        }
    }

    if (task_count > 0) {
        mf_task* last_task = &tasks[task_count - 1];
        last_task->inst_count = (u32)instr_count - last_task->start_inst;
        last_task->binding_offset = total_binding_count;
        last_task->binding_count = current_binding_count;
        for (u32 b = 0; b < current_binding_count; ++b) {
            bindings[total_binding_count].reg_idx = current_bindings[b].reg_idx;
            bindings[total_binding_count].byte_stride = current_bindings[b].byte_stride;
            bindings[total_binding_count].flags = current_bindings[b].is_reduction ? MF_BINDING_FLAG_REDUCTION : 0;
            total_binding_count++;
        }
    }
        
    prog->code = instrs;
    prog->tasks = tasks;
    prog->bindings = bindings;
    prog->meta.instruction_count = (u32)instr_count;
    prog->meta.task_count = task_count;
    prog->meta.binding_count = total_binding_count;

    u32 reduction_reg_count = 0;
    for (int r = 0; r < (int)prog->meta.tensor_count; ++r) {
        if (prog->tensor_flags[r] & MF_TENSOR_FLAG_REDUCTION) reduction_reg_count = MF_MAX_REGISTERS; 
    }

    prog->meta.reduction_scratch_size = reduction_reg_count;
    prog->meta.sync_scratch_size = needs_sync_scratch ? 1024 : 0; 

    return true;
}