#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/isa/mf_op_defs.h>
#include <mathflow/base/mf_shape.h>
#include <string.h>

#define MF_REPORT(node, msg, ...) \
    mf_compiler_diag_report(diag, (node)->loc, msg, ##__VA_ARGS__)

static bool shapes_match(const mf_type_info* a, const mf_type_info* b) {
    if (a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

static bool is_scalar(const mf_type_info* info) {
    return (info->ndim == 0 || (info->ndim == 1 && info->shape[0] == 1));
}

bool mf_pass_validate(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, mf_compiler_diag* diag) {
    bool success = true;

    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        if (node->type == MF_NODE_UNKNOWN) continue;

        u32 node_idx = (u32)(node - ir->nodes);
        mf_ir_node* inputs[4] = {0};
        
        // 1. Generic Arity & Type Validation
        for (u8 k = 0; k < meta->arity; ++k) {
            inputs[k] = mf_ir_find_input_by_name(ir, node_idx, meta->ports[k]);
            if (!inputs[k]) {
                MF_REPORT(node, "Validation Error: Missing required input port '%s' for node '%s' (%s)", 
                    meta->ports[k], node->id, meta->name);
                success = false;
                continue;
            }

            // Check if input dtype matches operation's mask
            u32 type_bit = (1 << inputs[k]->out_info.dtype);
            if (!(meta->input_mask & type_bit)) {
                MF_REPORT(node, "Type Mismatch: Input '%s' of node '%s' has invalid dtype. Operation '%s' expects mask 0x%X",
                    meta->ports[k], node->id, meta->name, meta->input_mask);
                success = false;
            }
        }

        // 2. Shape-Rule Specific Validation
        const mf_type_info* info1 = inputs[0] ? &inputs[0]->out_info : NULL;
        const mf_type_info* info2 = inputs[1] ? &inputs[1]->out_info : NULL;

        switch (meta->shape_rule) {
            case MF_SHAPE_BROADCAST:
                if (info1 && info2) {
                    if (!is_scalar(info1) && !is_scalar(info2) && !shapes_match(info1, info2)) {
                        MF_REPORT(node, "Shape Mismatch: Cannot broadcast inputs of node '%s'", node->id);
                        success = false;
                    }
                }
                break;

            case MF_SHAPE_SAME_AS_S1:
                if (info1 && !shapes_match(&node->out_info, info1)) {
                    MF_REPORT(node, "Shape Error: Output of '%s' must match Input 1", node->id);
                    success = false;
                }
                break;

            case MF_SHAPE_MATMUL:
                if (info1 && info2) {
                    if (info1->ndim < 2 || info2->ndim < 2) {
                        MF_REPORT(node, "MatMul Error: Inputs must be at least 2D in '%s'", node->id);
                        success = false;
                    } else if (info1->shape[1] != info2->shape[0]) {
                        MF_REPORT(node, "MatMul Error: Inner dimensions mismatch [%d] vs [%d] in '%s'", 
                            info1->shape[1], info2->shape[0], node->id);
                        success = false;
                    }
                }
                break;

            case MF_SHAPE_DOT:
                if (info1 && info2) {
                    if (info1->shape[info1->ndim-1] != info2->shape[info2->ndim-1]) {
                        MF_REPORT(node, "Dot Error: Last dimensions mismatch in '%s'", node->id);
                        success = false;
                    }
                }
                break;

            default:
                break;
        }
    }

    return success;
}