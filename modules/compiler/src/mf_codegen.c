#include "mf_compiler_internal.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static void parse_provider(const char* provider, u16* out_builtin_id, u8* out_builtin_axis) {
    if (!provider || provider[0] == '\0') {
        *out_builtin_id = MF_BUILTIN_NONE;
        *out_builtin_axis = 0;
        return;
    }

    if (strncmp(provider, "host.index", 10) == 0) {
        *out_builtin_id = MF_BUILTIN_INDEX;
        if (provider[10] == '.' && provider[11] >= '0' && provider[11] <= '9') {
            *out_builtin_axis = (u8)atoi(provider + 11);
        } else {
            *out_builtin_axis = 0;
        }
    } else if (strcmp(provider, "host.time") == 0) {
        *out_builtin_id = MF_BUILTIN_TIME;
        *out_builtin_axis = 0;
    } else if (strcmp(provider, "host.resolution") == 0) {
        *out_builtin_id = MF_BUILTIN_RESOLUTION;
        *out_builtin_axis = 0;
    } else if (strcmp(provider, "host.mouse") == 0) {
        *out_builtin_id = MF_BUILTIN_MOUSE;
        *out_builtin_axis = 0;
    } else {
        *out_builtin_id = MF_BUILTIN_NONE;
        *out_builtin_axis = 0;
    }
}

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

    for (size_t i = 0; i < sorted_count; ++i) {
        mf_ir_node* node = sorted[i];
        u32 node_idx = (u32)(node - ir->nodes); 
        u16 reg_idx = node->out_reg_idx;
        
        // Symbol Table
        if (node->id && strcmp(node->id, "unknown") != 0) {
            mf_bin_symbol* sym = &prog->symbols[current_symbol++];
            strncpy(sym->name, node->id, MF_MAX_SYMBOL_NAME - 1);
            sym->name[MF_MAX_SYMBOL_NAME - 1] = '\0';
            sym->name_hash = mf_fnv1a_hash(sym->name);
            sym->register_idx = reg_idx;
            if (node->provider) {
                strncpy(sym->provider, node->provider, MF_MAX_SYMBOL_NAME - 1);
                sym->provider[MF_MAX_SYMBOL_NAME - 1] = '\0';
                parse_provider(node->provider, &sym->builtin_id, &sym->builtin_axis);
            } else {
                sym->provider[0] = '\0';
                sym->builtin_id = MF_BUILTIN_NONE;
                sym->builtin_axis = 0;
            }
            sym->flags = (node->type == MF_NODE_INPUT) ? MF_SYMBOL_FLAG_INPUT : 
                         (node->type == MF_NODE_OUTPUT) ? MF_SYMBOL_FLAG_OUTPUT : 0;
        }

        // Tensor Descriptor
        mf_tensor* t_desc = &prog->tensors[reg_idx];
        
        // Register Aliasing Safety: Keep the info that requires the most memory.
        // This ensures the engine allocates enough space for the largest node that uses this register.
        mf_tensor tmp_current = { .info = t_desc->info };
        size_t current_size = mf_tensor_size_bytes(&tmp_current);
        size_t new_size = mf_tensor_size_bytes(&node->out_shape);
        
        // Always copy info if ndim was 0 (first time) or if the new one is larger.
        if (t_desc->info.ndim == 0 || new_size > current_size) {
            t_desc->info = node->out_shape.info;
        }

        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        
        mf_ir_node* s1 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[0]);
        mf_ir_node* s2 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[1]);
        mf_ir_node* s3 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[2]); 
        mf_ir_node* s4 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[3]);

        if (node->type == MF_NODE_CONST) {
            if (node->constant.buffer) {
                t_desc->buffer = node->constant.buffer;
                t_desc->byte_offset = node->constant.byte_offset;
            }
        }

        // Emit Instruction
        uint32_t start_instr_idx = (uint32_t)instr_count;
        bool emitted = false;

        if (node->type == MF_NODE_SIZE && s1) {
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
            emitted = false; 
        } else {
            mf_instruction* inst = &instrs[instr_count];
            memset(inst, 0, sizeof(mf_instruction));
            inst->dest_idx = reg_idx;
            inst->src1_idx = s1 ? s1->out_reg_idx : 0;
            inst->src2_idx = s2 ? s2->out_reg_idx : 0;
            inst->src3_idx = s3 ? s3->out_reg_idx : 0;
            inst->src4_idx = s4 ? s4->out_reg_idx : 0;

            memcpy(inst->strides, node->strides, sizeof(i32) * 4);

            if (inst->opcode == MF_OP_SUM && inst->strides[1] > 0) {
                inst->strides[0] = -1; // Reduction flag
            }

            if (meta->category == MF_OP_CAT_SPECIAL) {
                if (node->type == MF_NODE_INPUT && s1) {
                    inst->opcode = MF_OP_COPY; instr_count++; emitted = true;
                } else if (node->type == MF_NODE_OUTPUT || node->type == MF_NODE_COPY) {
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
                if (task_count > 0) {
                    tasks[task_count - 1].inst_count = start_instr_idx - tasks[task_count - 1].start_inst;
                }
                tasks[task_count].start_inst = start_instr_idx;
                
                if (is_global && s1) {
                    tasks[task_count].domain_reg = s1->out_reg_idx;
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

    if (task_count > 0) {
        tasks[task_count - 1].inst_count = (u32)instr_count - tasks[task_count - 1].start_inst;
    }

    prog->code = instrs;
    prog->tasks = tasks;
    prog->meta.instruction_count = (u32)instr_count;
    prog->meta.task_count = task_count;
    
    return true;
}