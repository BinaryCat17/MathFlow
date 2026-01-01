#include "mf_passes.h"
#include <mathflow/base/mf_log.h>
#include <string.h>

static void mark_domain(mf_graph_ir* ir, u32 node_idx, u32 domain_idx) {
    mf_ir_node* node = &ir->nodes[node_idx];
    
    // If already marked by a different domain, it might be a shared dependency.
    // We prioritize the "largest" or "most specific" domain, 
    // or we leave it as shared (domain_node_idx = UINT32_MAX).
    if (node->domain_node_idx != UINT32_MAX) {
        if (node->domain_node_idx != domain_idx) {
            // Check if it's a scalar or constant. Scalars are usually global/shared.
            bool is_scalar = (node->out_shape.info.ndim == 0);
            if (!is_scalar) {
                // If it's not scalar and used by multiple domains, 
                // it's a complex case. For now, mark as shared.
                node->domain_node_idx = UINT32_MAX; 
            }
        }
        return;
    }

    node->domain_node_idx = domain_idx;

    // Recurse to inputs
    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].dst_node_idx == node_idx) {
            mark_domain(ir, ir->links[i].src_node_idx, domain_idx);
        }
    }
}

bool mf_pass_domain_split(mf_graph_ir* ir, mf_compiler_diag* diag) {
    if (!ir) return false;

    // 1. Reset all domain indices
    for (size_t i = 0; i < ir->node_count; ++i) {
        ir->nodes[i].domain_node_idx = UINT32_MAX;
    }

    // 2. Find all Outputs and propagate their domain backwards
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].type == MF_NODE_OUTPUT) {
            mark_domain(ir, (u32)i, (u32)i);
        }
    }

    // 3. Any node still marked UINT32_MAX is either:
    //    - Not used by any Output (DCE candidate)
    //    - A global constant/input
    //    - Part of a cycle (should have been caught by topo sort)
    
    // For now, nodes without domain are treated as "Global" (domain 0 or similar)
    // but CodeGen expects a valid domain_reg. We can use the node itself if it's a constant.

    return true;
}
