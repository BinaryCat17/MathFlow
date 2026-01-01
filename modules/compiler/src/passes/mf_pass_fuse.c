#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_log.h>
#include <string.h>
#include <stdlib.h>

bool mf_pass_fuse(mf_graph_ir* ir, mf_compiler_diag* diag) {
    if (!ir) return false;

    // 1. Calculate use counts
    u32* use_count = calloc(ir->node_count, sizeof(u32));
    if (!use_count) return false;

    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].src_node_idx < ir->node_count) {
            use_count[ir->links[i].src_node_idx]++;
        }
    }

    bool changed = false;

    // 2. Look for (A * B) + C
    for (size_t i = 0; i < ir->node_count; ++i) {
        mf_ir_node* node = &ir->nodes[i];
        if (node->type != MF_NODE_ADD) continue;

        // Try both orderings: (Mul + C) and (C + Mul)
        for (int side = 0; side < 2; ++side) {
            u32 mul_port = side;
            u32 other_port = 1 - side;

            mf_ir_node* mul_node = find_input_source(ir, (u32)i, mul_port);
            if (mul_node && mul_node->type == MF_NODE_MUL && use_count[mul_node - ir->nodes] == 1) {
                
                u32 mul_idx = (u32)(mul_node - ir->nodes);
                mf_ir_node* m_s1 = find_input_source(ir, mul_idx, 0);
                mf_ir_node* m_s2 = find_input_source(ir, mul_idx, 1);
                mf_ir_node* other_input = find_input_source(ir, (u32)i, other_port);

                if (m_s1 && m_s2 && other_input) {
                    MF_LOG_DEBUG("Fusing MUL (%s) and ADD (%s) into FMA", mul_node->id, node->id);
                    
                    // Transform current ADD node into FMA
                    node->type = MF_NODE_FMA;
                    
                    // Update links to point to FMA
                    for (size_t l = 0; l < ir->link_count; ++l) {
                        mf_ir_link* link = &ir->links[l];
                        
                        // Link that was going to Mul -> now to Fma
                        if (link->dst_node_idx == mul_idx) {
                            link->dst_node_idx = (u32)i;
                            // dst_port (0 or 1) remains same for Fma
                        }
                        // Link that was going to Add.other_port -> now to Fma.port2
                        else if (link->dst_node_idx == (u32)i && link->dst_port == other_port) {
                            link->dst_port = 2;
                        }
                        // Link from Mul to Add -> delete
                        else if (link->src_node_idx == mul_idx && link->dst_node_idx == (u32)i) {
                            link->src_node_idx = UINT32_MAX;
                            link->dst_node_idx = UINT32_MAX;
                        }
                    }

                    mul_node->type = MF_NODE_UNKNOWN; 
                    changed = true;
                    break; 
                }
            }
        }
    }

    if (changed) {
        // Cleanup deleted links
        size_t write_idx = 0;
        for (size_t l = 0; l < ir->link_count; ++l) {
            if (ir->links[l].src_node_idx != UINT32_MAX) {
                ir->links[write_idx++] = ir->links[l];
            }
        }
        ir->link_count = write_idx;
    }

    free(use_count);
    return true;
}
