#include "mf_compiler_internal.h"
#include <mathflow/base/mf_json.h>
#include <string.h>

void mf_ir_parse_window_settings(const mf_json_value* root, mf_graph_ir* out_ir) {
    if (!root || root->type != MF_JSON_VAL_OBJECT) return;

    // Defaults
    strncpy(out_ir->app_title, "MathFlow Cartridge", MF_MAX_TITLE_NAME - 1);
    out_ir->window_width = 800;
    out_ir->window_height = 600;
    out_ir->vsync = 1;
    out_ir->resizable = 1;

    const mf_json_value* window = mf_json_get_field(root, "window");
    if (window && window->type == MF_JSON_VAL_OBJECT) {
        const mf_json_value* title = mf_json_get_field(window, "title");
        if (title && title->type == MF_JSON_VAL_STRING) strncpy(out_ir->app_title, title->as.s, MF_MAX_TITLE_NAME - 1);
        
        const mf_json_value* w = mf_json_get_field(window, "width");
        if (w && w->type == MF_JSON_VAL_NUMBER) out_ir->window_width = (u32)w->as.n;

        const mf_json_value* h = mf_json_get_field(window, "height");
        if (h && h->type == MF_JSON_VAL_NUMBER) out_ir->window_height = (u32)h->as.n;

        const mf_json_value* vsync = mf_json_get_field(window, "vsync");
        if (vsync && vsync->type == MF_JSON_VAL_BOOL) out_ir->vsync = vsync->as.b;

        const mf_json_value* fs = mf_json_get_field(window, "fullscreen");
        if (fs && fs->type == MF_JSON_VAL_BOOL) out_ir->fullscreen = fs->as.b;

        const mf_json_value* resizable = mf_json_get_field(window, "resizable");
        if (resizable && resizable->type == MF_JSON_VAL_BOOL) out_ir->resizable = resizable->as.b;
    }

    const mf_json_value* runtime = mf_json_get_field(root, "runtime");
    if (runtime && runtime->type == MF_JSON_VAL_OBJECT) {
        const mf_json_value* threads = mf_json_get_field(runtime, "threads");
        if (threads && threads->type == MF_JSON_VAL_NUMBER) out_ir->num_threads = (u32)threads->as.n;
    }
}

// --- Helper: Find Input Source ---
mf_ir_node* find_input_source(mf_graph_ir* ir, u32 dst_node_idx, u32 dst_port) {
    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].dst_node_idx == dst_node_idx && ir->links[i].dst_port == dst_port) {
            return &ir->nodes[ir->links[i].src_node_idx];
        }
    }
    return NULL;
}

mf_ir_node* mf_ir_find_input_by_name(mf_graph_ir* ir, u32 dst_node_idx, const char* port_name) {
    if (!port_name) return NULL;
    for (size_t i = 0; i < ir->link_count; ++i) {
        if (ir->links[i].dst_node_idx == dst_node_idx) {
            if (ir->links[i].dst_port_name && strcmp(ir->links[i].dst_port_name, port_name) == 0) {
                return &ir->nodes[ir->links[i].src_node_idx];
            }
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

static bool visit_node(sort_ctx* ctx, u32 node_idx) {
    if (ctx->visited[node_idx] == 2) return true;
    
    // Cycle detection
    if (ctx->visited[node_idx] == 1) {
        return false; 
    }
    
    ctx->visited[node_idx] = 1;

    // Visit dependencies
    for (size_t i = 0; i < ctx->ir->link_count; ++i) {
        if (ctx->ir->links[i].dst_node_idx == node_idx) {
            if (!visit_node(ctx, ctx->ir->links[i].src_node_idx)) return false;
        }
    }

    ctx->visited[node_idx] = 2;
    ctx->sorted_nodes[ctx->count++] = &ctx->ir->nodes[node_idx];
    return true;
}

mf_ir_node** mf_topo_sort(mf_graph_ir* ir, mf_arena* arena, size_t* out_count) {
    mf_ir_node** sorted = MF_ARENA_PUSH(arena, mf_ir_node*, ir->node_count);
    u8* visited = MF_ARENA_PUSH(arena, u8, ir->node_count);
    if (!sorted || !visited) return NULL;
    
    memset(visited, 0, ir->node_count);

    sort_ctx ctx = { .sorted_nodes = sorted, .visited = visited, .count = 0, .ir = ir };
    for (size_t i = 0; i < ir->node_count; ++i) {
        if (visited[i] == 0) {
            if (!visit_node(&ctx, (u32)i)) return NULL;
        }
    }
    
    if (out_count) *out_count = ctx.count;
    return sorted;
}
