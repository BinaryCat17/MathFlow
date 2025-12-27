#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/isa/mf_opcodes.h>
#include "mf_compiler_internal.h"
#include <string.h>
#include <stdio.h>

// Note: Logic moved to sub-modules (mf_json_parser.c, mf_semantics.c, mf_codegen.c)

// --- Compilation ---

mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena) {
    // 1. Sort
    size_t sorted_count = 0;
    mf_ir_node** sorted = mf_topo_sort(ir, arena, &sorted_count);

    // 2. Allocate Program
    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta.magic = MF_BINARY_MAGIC;
    prog->meta.version = MF_BINARY_VERSION;
    prog->meta.tensor_count = (u32)ir->node_count; // 1 output tensor per node
    
    // Count symbols (named nodes)
    u32 symbol_count = 0;
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].id && strcmp(ir->nodes[i].id, "unknown") != 0) {
            symbol_count++;
        }
    }
    prog->meta.symbol_count = symbol_count;
    prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, symbol_count);

    // Allocate Tensor Table (Descriptors)
    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, prog->meta.tensor_count);
    memset(prog->tensors, 0, sizeof(mf_tensor) * prog->meta.tensor_count);

    // 3. Instruction Generation & Tensor Init
    mf_instruction* instrs = MF_ARENA_PUSH(arena, mf_instruction, ir->node_count * 2);
    size_t instr_count = 0;
    u32 current_symbol = 0;

    for (size_t i = 0; i < sorted_count; ++i) {
        mf_ir_node* node = sorted[i];
        u32 node_idx = (u32)(node - ir->nodes); 
        
        // Assign Register (Tensor Index)
        node->out_reg_idx = (u16)node_idx; // Simple 1-to-1 mapping for now
        
        // Add to Symbol Table if it has a valid ID
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

        // If INPUT, copy constant data
        if (node->type == MF_NODE_INPUT) {
            *t_desc = node->constant; 
            node->out_shape = node->constant;
        } 
        else if (node->type == MF_NODE_MEMORY) {
            // Memory: Initial state comes from constant 'init'
            *t_desc = node->constant;
            node->out_shape = node->constant;
            // It acts as a constant/variable in the data section.
        }
        else {
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
            
            case MF_NODE_MEMORY:
                // Memory nodes do not generate compute instructions in the main pass.
                // Their update (State Transition) is handled in the "Append Memory Updates" pass below.
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

            case MF_NODE_MIX:
                if (s1 && s2 && s3) {
                    // Mix(a, b, t) = a + (b - a) * t
                    // s1=a, s2=b, s3=t
                    
                    // 1. Dest = Sub(b, a)
                    inst->opcode = MF_OP_SUB;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = s2->out_reg_idx;
                    inst->src2_idx = s1->out_reg_idx;
                    instr_count++;
                    
                    // 2. Dest = Mul(Dest, t)
                    inst = &instrs[instr_count];
                    inst->opcode = MF_OP_MUL;
                    inst->dest_idx = node->out_reg_idx;
                    inst->src1_idx = node->out_reg_idx;
                    inst->src2_idx = s3->out_reg_idx;
                    instr_count++;

                    // 3. Dest = Add(Dest, a)
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

    // 4. Append Memory Updates (State Transition)
    // We iterate over all nodes again to find Memory nodes and generate their update instructions.
    // This ensures updates happen AFTER all calculations.
    for (size_t i = 0; i < ir->node_count; ++i) {
        mf_ir_node* node = &ir->nodes[i];
        if (node->type == MF_NODE_MEMORY) {
            mf_ir_node* input = find_input_source(ir, (u32)(node - ir->nodes), 0);
            if (input) {
                mf_instruction* inst = &instrs[instr_count++];
                inst->opcode = MF_OP_COPY;
                inst->dest_idx = node->out_reg_idx;
                inst->src1_idx = input->out_reg_idx;
                inst->src2_idx = 0;
            }
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

    // 3. Symbol Table
    if (prog->meta.symbol_count > 0) {
        fwrite(prog->symbols, sizeof(mf_bin_symbol), prog->meta.symbol_count, f);
    }

    // 4. Tensor Metadata
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

    // 5. Tensor Data Blob
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
