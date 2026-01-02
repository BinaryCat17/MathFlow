#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/isa/mf_op_defs.h>
#include <string.h>

#define MF_REPORT(node, msg, ...) \
    mf_compiler_diag_report(diag, (node)->loc, msg, ##__VA_ARGS__)

bool mf_pass_validate(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, mf_compiler_diag* diag) {
    bool success = true;

    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        
        u32 node_idx = (u32)(node - ir->nodes);
        mf_ir_node* s1 = find_input_source(ir, node_idx, 0);
        mf_ir_node* s2 = find_input_source(ir, node_idx, 1);
        mf_ir_node* s3 = find_input_source(ir, node_idx, 2);

        mf_identity node_id = node->out_shape.info.identity;

        // 1. Check Identity Compatibility for Outputs
        // If this node is an Output node, it must match the expected identity of the sink.
        // Currently, we don't have explicit sink identity in IR, but we can check internal consistency.
        
        // Example: You cannot use a SPATIAL node as an input to a node that expects UNIFORM
        // (unless it's a reduction).
        if (meta->category != MF_OP_CAT_REDUCTION && meta->category != MF_OP_CAT_SPECIAL) {
            mf_ir_node* srcs[] = { s1, s2, s3 };
            for (int s = 0; s < 3; ++s) {
                if (srcs[s]) {
                    mf_identity src_id = srcs[s]->out_shape.info.identity;
                    if (src_id > node_id && node_id != MF_IDENTITY_UNKNOWN) {
                        MF_REPORT(node, "Identity Mismatch: Node '%s' (Identity: %d) cannot accept SPATIAL input from node %d", 
                            meta->name, node_id, (u32)(srcs[s] - ir->nodes));
                        success = false;
                    }
                }
            }
        }

        // 2. Check Domain Consistency
        // Nodes in the same task should ideally have the same domain or be uniform.
        if (node->domain_node_idx != UINT32_MAX) {
            mf_ir_node* dom_node = &ir->nodes[node->domain_node_idx];
            if (dom_node->out_shape.info.identity != MF_IDENTITY_SPATIAL) {
                // If domain is not spatial, then the node itself should not be spatial
                if (node_id == MF_IDENTITY_SPATIAL && node->type != MF_NODE_INDEX) {
                    MF_REPORT(node, "Domain Inconsistency: Node '%s' is SPATIAL but its domain is not.", meta->name);
                    success = false;
                }
            }
        }
    }

    return success;
}
