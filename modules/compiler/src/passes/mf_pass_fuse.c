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
        const char* add_ports[] = { "a", "b" };
        for (int side = 0; side < 2; ++side) {
            const char* mul_port_name = add_ports[side];
            const char* other_port_name = add_ports[1 - side];

            mf_ir_node* mul_node = mf_ir_find_input_by_name(ir, (u32)i, mul_port_name);
            if (mul_node && mul_node->type == MF_NODE_MUL && use_count[mul_node - ir->nodes] == 1) {
                
                u32 mul_idx = (u32)(mul_node - ir->nodes);
                mf_ir_node* m_s1 = mf_ir_find_input_by_name(ir, mul_idx, "a");
                mf_ir_node* m_s2 = mf_ir_find_input_by_name(ir, mul_idx, "b");
                mf_ir_node* other_input = mf_ir_find_input_by_name(ir, (u32)i, other_port_name);

                if (m_s1 && m_s2 && other_input) {
                    MF_LOG_DEBUG("Fusing MUL (%s) and ADD (%s) into FMA", mul_node->id, node->id);
                    
                    // Transform current ADD node into FMA
                    node->type = MF_NODE_FMA;
                    
                    // Update links to point to FMA
                    for (size_t l = 0; l < ir->link_count; ++l) {
                        mf_ir_link* link = &ir->links[l];
                        
                        // Link that was going to Mul -> now to Fma (ports "a" and "b" match)
                        if (link->dst_node_idx == mul_idx) {
                            link->dst_node_idx = (u32)i;
                            // dst_port and dst_port_name remain "a" or "b"
                        }
                        // Link that was going to Add.other_port -> now to Fma.port "c"
                        else if (link->dst_node_idx == (u32)i && strcmp(link->dst_port_name, other_port_name) == 0) {
                            link->dst_port_name = "c";
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
