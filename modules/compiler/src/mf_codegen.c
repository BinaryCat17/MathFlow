#include "mf_compiler_internal.h"
#include <string.h>

// --- Topological Sort Helpers ---

typedef struct {
    mf_ir_node** sorted_nodes; 
    u8* visited;
    size_t count;
    mf_graph_ir* ir;
} sort_ctx;

static void visit_node(sort_ctx* ctx, u32 node_idx) {
    if (ctx->visited[node_idx] == 2) return;
    
    // Cycle detection
    if (ctx->visited[node_idx] == 1) {
        // If we hit a node currently being visited, it's a cycle.
        // However, if the current node is a MEMORY node, it breaks the cycle naturally 
        // because its output does not depend on its input within the SAME frame.
        return; 
    }
    
    ctx->visited[node_idx] = 1;

    // Visit dependencies
    // For Memory nodes, we do NOT visit dependencies during Topo Sort.
    // They act as inputs for the current frame.
    if (ctx->ir->nodes[node_idx].type != MF_NODE_MEMORY) {
        for (size_t i = 0; i < ctx->ir->link_count; ++i) {
            if (ctx->ir->links[i].dst_node_idx == node_idx) {
                visit_node(ctx, ctx->ir->links[i].src_node_idx);
            }
        }
    }

    ctx->visited[node_idx] = 2;
    ctx->sorted_nodes[ctx->count++] = &ctx->ir->nodes[node_idx];
}

mf_ir_node** mf_topo_sort(mf_graph_ir* ir, mf_arena* arena, size_t* out_count) {
    mf_ir_node** sorted = MF_ARENA_PUSH(arena, mf_ir_node*, ir->node_count);
    u8* visited = MF_ARENA_PUSH(arena, u8, ir->node_count);
    memset(visited, 0, ir->node_count);

    sort_ctx ctx = { .sorted_nodes = sorted, .visited = visited, .count = 0, .ir = ir };
    for (size_t i = 0; i < ir->node_count; ++i) visit_node(&ctx, (u32)i);
    
    if (out_count) *out_count = ctx.count;
    return sorted;
}
