#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <string.h>
#include <stdio.h>

// --- Metadata Lookups ---

static mf_node_type get_node_type(const char* type_str) {
    if (!type_str) return MF_NODE_UNKNOWN;
    for (int i = 1; i < MF_NODE_COUNT; ++i) {
        if (strcmp(type_str, MF_OP_METADATA[i].name) == 0) return (mf_node_type)i;
    }
    return MF_NODE_UNKNOWN;
}

static u32 get_port_index(mf_node_type type, const char* port_name) {
    if (!port_name || type >= MF_NODE_COUNT) return 0;
    const mf_op_metadata* meta = &MF_OP_METADATA[type];
    for (u32 i = 0; i < 4; ++i) {
        if (meta->ports[i] && strcmp(meta->ports[i], port_name) == 0) return i;
    }
    return 0;
}

// --- Helpers ---

static void parse_const_tensor(const mf_json_value* val, const mf_json_value* node_data, mf_type_info* info, void** out_data, mf_arena* arena) {
    if (!val || !info || !out_data) return;

    // Default to scalar F32 if no metadata provided
    mf_dtype dtype = MF_DTYPE_F32;
    int32_t shape[MF_MAX_DIMS] = {0};
    uint8_t ndim = 0;

    const mf_json_value* meta = mf_json_get_field(node_data, "meta");
    if (meta) {
        const mf_json_value* j_dtype = mf_json_get_field(meta, "dtype");
        if (j_dtype) dtype = mf_dtype_from_str(j_dtype->as.s);

        const mf_json_value* j_shape = mf_json_get_field(meta, "shape");
        if (j_shape && j_shape->type == MF_JSON_VAL_ARRAY) {
            ndim = (uint8_t)j_shape->as.array.count;
            for (int i = 0; i < ndim; ++i) {
                shape[i] = (int32_t)j_shape->as.array.items[i].as.n;
            }
        }
    } else if (val->type == MF_JSON_VAL_ARRAY) {
        ndim = 1;
        shape[0] = (int32_t)val->as.array.count;
    }

    mf_type_info_init_contiguous(info, dtype, shape, ndim);
    size_t count = mf_shape_calc_count(shape, ndim);
    size_t bytes = count * mf_dtype_size(dtype);
    
    *out_data = MF_ARENA_PUSH(arena, uint8_t, bytes);
    
    if (val->type == MF_JSON_VAL_ARRAY) {
        for (size_t i = 0; i < val->as.array.count && i < count; ++i) {
            mf_json_value* item = &val->as.array.items[i];
            if (dtype == MF_DTYPE_F32) ((f32*)*out_data)[i] = (f32)item->as.n;
            else if (dtype == MF_DTYPE_I32) ((i32*)*out_data)[i] = (i32)item->as.n;
            else if (dtype == MF_DTYPE_U8) ((u8*)*out_data)[i] = (u8)item->as.n;
        }
    } else if (val->type == MF_JSON_VAL_NUMBER) {
        if (dtype == MF_DTYPE_F32) ((f32*)*out_data)[0] = (f32)val->as.n;
        else if (dtype == MF_DTYPE_I32) ((i32*)*out_data)[0] = (i32)val->as.n;
        else if (dtype == MF_DTYPE_U8) ((u8*)*out_data)[0] = (u8)val->as.n;
    }
}

static bool lower_node(const mf_json_value* node_val, mf_ir_node* ir_node, mf_arena* arena, mf_compiler_diag* diag) {
    // ... search for usage of constant ...
    // I need to read more of this file to find where parse_const_tensor is called.
}

static void _parse_provider(const char* provider, u16* out_builtin_id, u8* out_builtin_axis) {
    if (!provider || provider[0] == '\0') {
        *out_builtin_id = MF_BUILTIN_NONE;
        *out_builtin_axis = 0;
        return;
    }

    if (strncmp(provider, "host.index", 10) == 0) {
        *out_builtin_id = MF_BUILTIN_INDEX;
        if (provider[10] == '.' && provider[11] >= '0' && provider[11] <= '9') {
            *out_builtin_axis = (u8)atoi(provider + 11);
        } else {
            *out_builtin_axis = 0;
        }
    } else {
        *out_builtin_id = MF_BUILTIN_NONE;
        *out_builtin_axis = 0;
    }
}

static bool parse_node_attributes(mf_ir_node* dst, const mf_json_value* data, const char* base_path, mf_arena* arena, mf_compiler_diag* diag) {
    if (!data) return true;

    switch (dst->type) {
        case MF_NODE_INPUT:
        case MF_NODE_OUTPUT: {
            const mf_json_value* v_shape = mf_json_get_field(data, "shape");
            const mf_json_value* v_dtype = mf_json_get_field(data, "dtype");
            const mf_json_value* v_provider = mf_json_get_field(data, "provider");
            const mf_json_value* v_readonly = mf_json_get_field(data, "readonly");
            const mf_json_value* v_persistent = mf_json_get_field(data, "persistent");
            
            if (v_provider && v_provider->type == MF_JSON_VAL_STRING) {
                dst->provider = mf_arena_strdup(arena, v_provider->as.s);
                _parse_provider(dst->provider, &dst->builtin_id, &dst->builtin_axis);
            }

            if (v_readonly && v_readonly->type == MF_JSON_VAL_BOOL && v_readonly->as.b) {
                dst->resource_flags |= MF_RESOURCE_FLAG_READONLY;
            }
            if (v_persistent && v_persistent->type == MF_JSON_VAL_BOOL && v_persistent->as.b) {
                dst->resource_flags |= MF_RESOURCE_FLAG_PERSISTENT;
            }

            if (dst->type == MF_NODE_INPUT && (!v_shape || v_shape->type != MF_JSON_VAL_ARRAY)) {
                mf_compiler_diag_report(diag, dst->loc, "Input node '%s': missing or invalid 'shape'", dst->id);
                return false;
            }
            
            dst->const_info.dtype = v_dtype ? mf_dtype_from_str(v_dtype->as.s) : MF_DTYPE_F32;
            
            if (v_shape && v_shape->type == MF_JSON_VAL_ARRAY) {
                dst->const_info.ndim = (uint8_t)v_shape->as.array.count;
                if (dst->const_info.ndim > MF_MAX_DIMS) dst->const_info.ndim = MF_MAX_DIMS;
                for(int k=0; k<dst->const_info.ndim; ++k) {
                     const mf_json_value* dim = &v_shape->as.array.items[k];
                     if (dim->type == MF_JSON_VAL_NUMBER) {
                         int d = (int)dim->as.n;
                         dst->const_info.shape[k] = (d < 0) ? -1 : d;
                     }
                }
                mf_shape_calc_strides(&dst->const_info);
                // For Output nodes, we also copy this to out_info initially
                if (dst->type == MF_NODE_OUTPUT) {
                    dst->out_info = dst->const_info;
                }
            }
            break;
        }
        case MF_NODE_CONST: {
            const mf_json_value* v_val = mf_json_get_field(data, "value");
            if (v_val) parse_const_tensor(v_val, data, &dst->const_info, &dst->const_data, arena);
            break;
        }
        case MF_NODE_CALL: {
            const mf_json_value* v_path = mf_json_get_field(data, "path");
            if (v_path && v_path->type == MF_JSON_VAL_STRING) {
                if (base_path) {
                    char* dir = mf_path_get_dir(base_path, arena);
                    dst->sub_graph_path = mf_path_join(dir, v_path->as.s, arena);
                } else {
                    dst->sub_graph_path = mf_arena_strdup(arena, v_path->as.s);
                }
            }
            break;
        }
        default: break;
    }
    return true;
}

// --- Main Pass ---

bool mf_pass_lower(mf_ast_graph* ast, mf_graph_ir* out_ir, mf_arena* arena, const char* base_path, mf_compiler_diag* diag) {
    if (!ast) {
        MF_REPORT(diag, NULL, "Lowering Pass: AST is NULL");
        return false;
    }

    memset(out_ir, 0, sizeof(mf_graph_ir));

    // --- Process Root App Settings (Cartridge Metadata) ---
    mf_ir_parse_window_settings(ast->root, out_ir);

    out_ir->node_count = ast->node_count;
    out_ir->node_cap = ast->node_count;
    out_ir->nodes = MF_ARENA_PUSH(arena, mf_ir_node, ast->node_count);
    memset(out_ir->nodes, 0, sizeof(mf_ir_node) * ast->node_count);

    mf_str_map map;
    mf_map_init(&map, ast->node_count * 2, arena);

    // 1. Process Nodes
    for (size_t i = 0; i < ast->node_count; ++i) {
        mf_ast_node* src = &ast->nodes[i];
        mf_ir_node* dst = &out_ir->nodes[i];

        dst->loc.file = base_path ? mf_arena_strdup(arena, base_path) : "unknown";
        dst->loc.line = src->loc.line;
        dst->loc.column = src->loc.column;

        dst->id = mf_arena_strdup(arena, src->id);
        mf_map_put(&map, dst->id, i);

        dst->type = get_node_type(src->type);
        if (dst->type == MF_NODE_UNKNOWN) {
            // Explicit Import System: Search for <type>.json in imports
            bool found = false;
            
            // 1. Search in local imports
            for (size_t k = 0; k < ast->import_count; ++k) {
                const char* import_path = ast->imports[k];
                char* full_path;
                if (base_path) {
                    char* dir = mf_path_get_dir(base_path, arena);
                    full_path = mf_path_join(dir, import_path, arena);
                } else {
                    full_path = (char*)import_path;
                }
                
                char* file_path = mf_arena_sprintf(arena, "%s/%s.json", full_path, src->type);
                if (mf_file_exists(file_path)) {
                    dst->type = MF_NODE_CALL;
                    dst->sub_graph_path = file_path;
                    found = true;
                    break;
                }
            }
            
            // 2. Search in Global Prelude (assets/lib)
            if (!found) {
                char* file_path = mf_arena_sprintf(arena, "assets/lib/%s.json", src->type);
                if (mf_file_exists(file_path)) {
                    dst->type = MF_NODE_CALL;
                    dst->sub_graph_path = file_path;
                    found = true;
                }
            }

            if (!found) {
                mf_compiler_diag_report(diag, dst->loc, "Unknown node type '%s' (not a built-in and not found in imports)", src->type);
                return false;
            }
        }

        if (!parse_node_attributes(dst, src->data, base_path, arena, diag)) return false;
    }

    // 2. Process Domains (Must happen after all nodes are in the map)
    for (size_t i = 0; i < ast->node_count; ++i) {
        mf_ast_node* src = &ast->nodes[i];
        mf_ir_node* dst = &out_ir->nodes[i];
        dst->domain_node_idx = UINT32_MAX; // Default

        if (src->data) {
            const mf_json_value* v_domain = mf_json_get_field(src->data, "domain");
            if (v_domain && v_domain->type == MF_JSON_VAL_STRING) {
                u32 dom_idx;
                if (mf_map_get(&map, v_domain->as.s, &dom_idx)) {
                    dst->domain_node_idx = dom_idx;
                } else {
                    mf_compiler_diag_report(diag, dst->loc, "Domain node '%s' not found for node '%s'", v_domain->as.s, dst->id);
                    return false;
                }
            }
        }
    }

    // 3. Process Links
    out_ir->link_count = ast->link_count;
    out_ir->link_cap = ast->link_count;
    out_ir->links = MF_ARENA_PUSH(arena, mf_ir_link, ast->link_count);

    for (size_t i = 0; i < ast->link_count; ++i) {
        mf_ast_link* l_src = &ast->links[i];
        mf_ir_link* l_dst = &out_ir->links[i];
        
        // Source
        if (!mf_map_get(&map, l_src->src, &l_dst->src_node_idx)) {
            mf_source_loc loc = {base_path, l_src->loc.line, l_src->loc.column};
            mf_compiler_diag_report(diag, loc, "Link source '%s' not found", l_src->src);
            return false;
        }
        mf_ir_node* src_node = &out_ir->nodes[l_dst->src_node_idx];
        l_dst->src_port_name = l_src->src_port ? mf_arena_strdup(arena, l_src->src_port) : "out";
        if (src_node->type != MF_NODE_CALL) {
            l_dst->src_port = get_port_index(src_node->type, l_src->src_port);
        }

        // Dest
        if (!mf_map_get(&map, l_src->dst, &l_dst->dst_node_idx)) {
            mf_source_loc loc = {base_path, l_src->loc.line, l_src->loc.column};
            mf_compiler_diag_report(diag, loc, "Link dst '%s' not found", l_src->dst);
            return false;
        }
        mf_ir_node* dst_node = &out_ir->nodes[l_dst->dst_node_idx];
        l_dst->dst_port_name = l_src->dst_port ? mf_arena_strdup(arena, l_src->dst_port) : "in";
        if (dst_node->type != MF_NODE_CALL) {
            l_dst->dst_port = get_port_index(dst_node->type, l_src->dst_port);
        }
    }

    return true;
}