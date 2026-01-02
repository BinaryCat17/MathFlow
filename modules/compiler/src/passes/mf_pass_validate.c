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
        
        u32 node_idx = (u32)(node - ir->nodes);
        mf_ir_node* s1 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[0]);
        mf_ir_node* s2 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[1]);
        mf_ir_node* s3 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[2]);

        const mf_type_info* info1 = s1 ? &s1->out_info : NULL;
        const mf_type_info* info2 = s2 ? &s2->out_info : NULL;
        const mf_type_info* info3 = s3 ? &s3->out_info : NULL;

        switch (meta->shape_rule) {
            case MF_SHAPE_BROADCAST:
                if (s1 && s2) {
                    // Basic broadcast validation: shapes must match or one must be scalar
                    if (!is_scalar(info1) && !is_scalar(info2) && !shapes_match(info1, info2)) {
                        MF_REPORT(node, "Shape Mismatch: Cannot broadcast %s and %s in '%s'", 
                            node->id, s1->id, s2->id);
                        success = false;
                    }
                }
                break;

            case MF_SHAPE_SAME_AS_S1:
                if (s1 && !shapes_match(&node->out_info, info1)) {
                    MF_REPORT(node, "Shape Error: Output shape does not match Input 1 in '%s'", node->id);
                    success = false;
                }
                break;

            case MF_SHAPE_MATMUL:
                if (s1 && s2) {
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
                if (s1 && s2) {
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
