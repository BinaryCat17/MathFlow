#include "mf_compiler_internal.h"
#include <string.h>

// --- Helper: Find Input Source ---
mf_ir_node* find_input_source(mf_graph_ir* ir, u32 dst_node_idx, u32 dst_port) {
    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].dst_node_idx == dst_node_idx && ir->links[i].dst_port == dst_port) {
            return &ir->nodes[ir->links[i].src_node_idx];
        }
    }
    return NULL;
}

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
        return; 
    }
    
    ctx->visited[node_idx] = 1;

    // Visit dependencies
    for (size_t i = 0; i < ctx->ir->link_count; ++i) {
        if (ctx->ir->links[i].dst_node_idx == node_idx) {
            visit_node(ctx, ctx->ir->links[i].src_node_idx);
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
