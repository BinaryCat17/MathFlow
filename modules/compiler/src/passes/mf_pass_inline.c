#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <string.h>

// Forward declaration of the parsing entry point needed for recursive loading
// Note: We use the high-level loading function that returns IR
bool mf_compile_load_json_ir(const char* json_path, mf_graph_ir* out_ir, mf_arena* arena, mf_compiler_diag* diag);

// --- Expansion Logic (Copied and cleaned from old parser) ---

static bool needs_expansion(mf_graph_ir* ir) {
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (ir->nodes[i].type == MF_NODE_CALL) return true;
    }
    return false;
}

static bool expand_graph_step(mf_graph_ir* src, mf_graph_ir* dst, mf_arena* arena, mf_compiler_diag* diag) {
    typedef struct LNode { mf_ir_node n; struct LNode* next; } LNode;
    typedef struct LLink { mf_ir_link l; struct LLink* next; } LLink;
    
    LNode* head_node = NULL;
    LNode* tail_node = NULL;
    size_t new_node_count = 0;

    LLink* head_link = NULL;
    LLink* tail_link = NULL;
    size_t new_link_count = 0;

    #define APPEND_NODE(node_val) { \
        LNode* ln = MF_ARENA_PUSH(arena, LNode, 1); \
        ln->n = node_val; ln->next = NULL; \
        if (tail_node) tail_node->next = ln; else head_node = ln; \
        tail_node = ln; \
        new_node_count++; \
    }

    #define APPEND_LINK(link_val) { \
        LLink* ll = MF_ARENA_PUSH(arena, LLink, 1); \
        ll->l = link_val; ll->next = NULL; \
        if (tail_link) tail_link->next = ll; else head_link = ll; \
        tail_link = ll; \
        new_link_count++; \
    }

    mf_str_map global_map;
    mf_map_init(&global_map, 4096, arena); 

    mf_str_map port_map;
    mf_map_init(&port_map, 1024, arena);

    u32 current_idx = 0;

    for (size_t i = 0; i < src->node_count; ++i) {
        mf_ir_node* node = &src->nodes[i];
        
        if (node->type == MF_NODE_CALL) {
            if (!node->sub_graph_path) continue;
            
            mf_graph_ir child_ir;
            // Recursive load!
            if (!mf_compile_load_json_ir(node->sub_graph_path, &child_ir, arena, diag)) {
                  // Error already reported by child loader
                 continue;
            }

            const char** child_raw_ids = MF_ARENA_PUSH(arena, const char*, child_ir.node_count);
            for (size_t k = 0; k < child_ir.node_count; ++k) child_raw_ids[k] = child_ir.nodes[k].id;

            u32 input_count = 0;
            u32 output_count = 0;
            const char* last_input_raw_id = NULL;
            const char* last_output_raw_id = NULL;

            // Inheritance context
            u32 parent_domain_idx = node->domain_node_idx;
            u32 mapped_domain_idx = UINT32_MAX;
            if (parent_domain_idx != UINT32_MAX) {
                const char* dom_id = src->nodes[parent_domain_idx].id;
                mf_map_get(&global_map, dom_id, &mapped_domain_idx);
            }

            u32* child_to_dst_map = MF_ARENA_PUSH(arena, u32, child_ir.node_count);
            for (size_t k = 0; k < child_ir.node_count; ++k) child_to_dst_map[k] = UINT32_MAX;

            // First pass: Allocate IDs and indices
            for (size_t k = 0; k < child_ir.node_count; ++k) {
                mf_ir_node* c_node = &child_ir.nodes[k];
                if (c_node->type == MF_NODE_OUTPUT) continue;
                
                const char* raw_id = child_raw_ids[k];
                char* new_id = mf_arena_sprintf(arena, "%s::%s", node->id, raw_id);
                c_node->id = new_id;
                
                child_to_dst_map[k] = current_idx;
                mf_map_put(&global_map, new_id, current_idx);
                current_idx++;
                
                if (c_node->type == MF_NODE_INPUT) {
                    char port_key[128];
                    snprintf(port_key, 128, "%s:i:%s", node->id, raw_id); 
                    mf_map_put(&port_map, mf_arena_strdup(arena, port_key), child_to_dst_map[k]); 
                    input_count++;
                    last_input_raw_id = raw_id;
                }
            }

            // Second pass: Map domains and Append
            for (size_t k = 0; k < child_ir.node_count; ++k) {
                mf_ir_node* c_node = &child_ir.nodes[k];
                if (c_node->type == MF_NODE_OUTPUT) {
                     // Handle output port registration (no node added)
                     u32 provider_node_idx = 0;
                     bool found = false;
                     for (size_t l = 0; l < child_ir.link_count; ++l) {
                         if (child_ir.links[l].dst_node_idx == (u32)k) {
                             provider_node_idx = child_ir.links[l].src_node_idx;
                             found = true;
                             break;
                         }
                     }
                     if (found) {
                         char port_key[128];
                         snprintf(port_key, 128, "%s:o:%s", node->id, child_raw_ids[k]);
                         const char* provider_raw_id = child_raw_ids[provider_node_idx];
                         char* provider_id = mf_arena_sprintf(arena, "%s::%s", node->id, provider_raw_id);
                         mf_map_put_ptr(&port_map, mf_arena_strdup(arena, port_key), provider_id);
                         output_count++;
                         last_output_raw_id = child_raw_ids[k];
                     }
                     continue;
                }

                // Domain Mapping
                if (c_node->domain_node_idx != UINT32_MAX) {
                    // Map domain from child IR to new indices
                    c_node->domain_node_idx = child_to_dst_map[c_node->domain_node_idx];
                } else {
                    // Inherit from parent Call node
                    c_node->domain_node_idx = mapped_domain_idx;
                }

                APPEND_NODE(*c_node);
            }

            // Register default ports if unambiguous
            if (input_count == 1) {
                char port_key[128];
                snprintf(port_key, 128, "%s:i:default", node->id);
                u32 input_idx;
                char last_input_id[256];
                snprintf(last_input_id, 256, "%s::%s", node->id, last_input_raw_id);
                if (mf_map_get(&global_map, last_input_id, &input_idx)) {
                    mf_map_put(&port_map, mf_arena_strdup(arena, port_key), input_idx);
                    MF_LOG_DEBUG("Inline: Registered default input for '%s' -> %s", node->id, last_input_raw_id);
                }
            }
            if (output_count == 1) {
                char port_key[128];
                snprintf(port_key, 128, "%s:o:default", node->id);
                void* provider_id = NULL;
                char port_search[128];
                snprintf(port_search, 128, "%s:o:%s", node->id, last_output_raw_id);
                if (mf_map_get_ptr(&port_map, port_search, &provider_id)) {
                    mf_map_put_ptr(&port_map, mf_arena_strdup(arena, port_key), provider_id);
                    MF_LOG_DEBUG("Inline: Registered default output for '%s' -> %s", node->id, last_output_raw_id);
                }
            }

            for (size_t k = 0; k < child_ir.link_count; ++k) {
                mf_ir_link l = child_ir.links[k];
                if (child_ir.nodes[l.dst_node_idx].type == MF_NODE_OUTPUT) continue;

                const char* src_id = child_ir.nodes[l.src_node_idx].id;
                const char* dst_id = child_ir.nodes[l.dst_node_idx].id;
                
                u32 new_src_idx, new_dst_idx;
                if (mf_map_get(&global_map, src_id, &new_src_idx) && 
                    mf_map_get(&global_map, dst_id, &new_dst_idx)) 
                {
                    l.src_node_idx = new_src_idx;
                    l.dst_node_idx = new_dst_idx;
                    APPEND_LINK(l);
                }
            }
        } 
        else {
            mf_map_put(&global_map, node->id, current_idx);
            APPEND_NODE(*node);
            current_idx++;
        }
    }

    for (size_t i = 0; i < src->link_count; ++i) {
        mf_ir_link l = src->links[i];
        mf_ir_node* src_node = &src->nodes[l.src_node_idx];
        mf_ir_node* dst_node = &src->nodes[l.dst_node_idx];

        u32 final_src_idx = 0;
        u32 final_dst_idx = 0;
        bool drop_link = false;

        if (src_node->type == MF_NODE_CALL) {
            char key[128];
            snprintf(key, 128, "%s:o:%s", src_node->id, l.src_port_name ? l.src_port_name : "unknown");
            
            void* resolved_ptr = NULL;
            if (mf_map_get_ptr(&port_map, key, &resolved_ptr)) {
                const char* provider_id = (const char*)resolved_ptr;
                if (!mf_map_get(&global_map, provider_id, &final_src_idx)) {
                    MF_LOG_DEBUG("Inline: Could not find provider '%s' in global_map", provider_id);
                    drop_link = true;
                }
                l.src_port = 0; 
            } else {
                MF_LOG_DEBUG("Inline: Could not find port key '%s' in port_map", key);
                drop_link = true;
            }
        } else {
            mf_map_get(&global_map, src_node->id, &final_src_idx);
        }

        if (dst_node->type == MF_NODE_CALL) {
            char key[128];
            snprintf(key, 128, "%s:i:%s", dst_node->id, l.dst_port_name ? l.dst_port_name : "unknown");
            if (!mf_map_get(&port_map, key, &final_dst_idx)) {
                MF_LOG_DEBUG("Inline: Could not find port key '%s' in port_map", key);
                drop_link = true;
            }
            else {
                l.dst_port = 0;
                l.dst_port_name = "out";
            }
        } else {
            mf_map_get(&global_map, dst_node->id, &final_dst_idx);
        }

        if (!drop_link) {
            l.src_node_idx = final_src_idx;
            l.dst_node_idx = final_dst_idx;
            APPEND_LINK(l);
        }
    }

    dst->node_count = new_node_count;
    dst->nodes = MF_ARENA_PUSH(arena, mf_ir_node, new_node_count);
    size_t ni = 0;
    for (LNode* cur = head_node; cur; cur = cur->next) {
        dst->nodes[ni++] = cur->n;
    }

    dst->link_count = new_link_count;
    dst->links = MF_ARENA_PUSH(arena, mf_ir_link, new_link_count);
    size_t li = 0;
    for (LLink* cur = head_link; cur; cur = cur->next) {
        dst->links[li++] = cur->l;
    }

    return true;
}

bool mf_pass_inline(mf_graph_ir* ir, mf_arena* arena, mf_compiler_diag* diag) {
    if (!ir) {
        MF_REPORT(diag, NULL, "Inline Pass: IR is NULL");
        return false;
    }

    mf_graph_ir current_ir = *ir;
    
    // Iteratively expand until no Calls remain
    for (int pass = 0; pass < 10; ++pass) {
        if (!needs_expansion(&current_ir)) {
            *ir = current_ir;
            return true;
        }
        
        mf_graph_ir next_ir;
        if (!expand_graph_step(&current_ir, &next_ir, arena, diag)) {
            MF_REPORT(diag, NULL, "Inline Pass: Expansion step failed");
            return false;
        }
        current_ir = next_ir;
    }
    
    { mf_source_loc loc = {0}; mf_compiler_diag_report(diag, loc, "Inline pass failed: Max recursion depth reached."); }
    return false;
}
