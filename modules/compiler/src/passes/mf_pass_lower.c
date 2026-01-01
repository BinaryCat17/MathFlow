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
    for (u32 i = 0; i < 3; ++i) {
        if (meta->ports[i] && strcmp(meta->ports[i], port_name) == 0) return i;
    }
    return 0;
}

// --- Helpers ---

static void parse_const_tensor(const mf_json_value* val, const mf_json_value* node_data, mf_tensor* t, mf_arena* arena) {
    mf_dtype target_dtype = MF_DTYPE_F32;
    if (node_data) {
        const mf_json_value* v_dt = mf_json_get_field(node_data, "dtype");
        if (v_dt && v_dt->type == MF_JSON_VAL_STRING) target_dtype = mf_dtype_from_str(v_dt->as.s);
    }

    if (val->type == MF_JSON_VAL_NUMBER) {
        t->info.dtype = target_dtype;
        t->info.ndim = 0;
        size_t bytes = mf_dtype_size(target_dtype);
        mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
        void* mem = MF_ARENA_PUSH(arena, u8, bytes);
        mf_buffer_init_view(buf, mem, bytes);
        t->buffer = buf;
        t->byte_offset = 0;
        if (target_dtype == MF_DTYPE_F32) *((f32*)mem) = (f32)val->as.n;
        else if (target_dtype == MF_DTYPE_I32) *((int32_t*)mem) = (int32_t)val->as.n;
        else if (target_dtype == MF_DTYPE_U8) *((u8*)mem) = (u8)val->as.n;
    } 
    else if (val->type == MF_JSON_VAL_BOOL) {
        t->info.dtype = MF_DTYPE_U8;
        t->info.ndim = 0;
        size_t bytes = sizeof(u8);
        mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
        void* mem = MF_ARENA_PUSH(arena, u8, bytes);
        mf_buffer_init_view(buf, mem, bytes);
        t->buffer = buf;
        t->byte_offset = 0;
        *((u8*)mem) = (u8)(val->as.b ? 1 : 0);
    }
    else if (val->type == MF_JSON_VAL_STRING) {
        if (target_dtype == MF_DTYPE_I32) {
            t->info.dtype = MF_DTYPE_I32;
            t->info.ndim = 0;
            size_t bytes = sizeof(int32_t);
            mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            mf_buffer_init_view(buf, mem, bytes);
            t->buffer = buf;
            t->byte_offset = 0;
            *((int32_t*)mem) = (int32_t)mf_fnv1a_hash(val->as.s);
        } else {
            size_t cp_count = mf_utf8_to_utf32(val->as.s, NULL, 0);
            t->info.dtype = MF_DTYPE_F32;
            t->info.ndim = 1;
            t->info.shape[0] = (int32_t)cp_count;
            mf_shape_calc_strides(&t->info);

            size_t bytes = cp_count * sizeof(f32);
            mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            mf_buffer_init_view(buf, mem, bytes);
            t->buffer = buf;
            t->byte_offset = 0;
            
            u32* tmp = malloc(cp_count * sizeof(u32));
            mf_utf8_to_utf32(val->as.s, tmp, cp_count);
            f32* dst = (f32*)mem;
            for(size_t i=0; i<cp_count; ++i) dst[i] = (f32)tmp[i];
            free(tmp);
        }
    }
    else if (val->type == MF_JSON_VAL_ARRAY) {
        int count = (int)val->as.array.count;
        if (count == 0) return;
        const mf_json_value* first = &val->as.array.items[0];
        const mf_json_value* v_shape = node_data ? mf_json_get_field(node_data, "shape") : NULL;

        if (first->type == MF_JSON_VAL_NUMBER) {
            t->info.dtype = MF_DTYPE_F32;
            if (v_shape && v_shape->type == MF_JSON_VAL_ARRAY) {
                t->info.ndim = (uint8_t)v_shape->as.array.count;
                for(int k=0; k<t->info.ndim && k<MF_MAX_DIMS; ++k) {
                    t->info.shape[k] = (int32_t)v_shape->as.array.items[k].as.n;
                }
            } else {
                t->info.ndim = 1;
                t->info.shape[0] = count;
            }
            mf_shape_calc_strides(&t->info);
            
            size_t bytes = count * sizeof(f32);
            mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            mf_buffer_init_view(buf, mem, bytes);
            t->buffer = buf;
            t->byte_offset = 0;
            
            f32* data = (f32*)mem;
            for(int i=0; i<count; ++i) {
                const mf_json_value* item = &val->as.array.items[i];
                if (item->type == MF_JSON_VAL_NUMBER) data[i] = (f32)item->as.n;
                else data[i] = 0.0f;
            }
        }
        else if (first->type == MF_JSON_VAL_STRING) {
            t->info.dtype = MF_DTYPE_I32;
            t->info.ndim = 1;
            t->info.shape[0] = count;
            mf_shape_calc_strides(&t->info);

            size_t bytes = count * sizeof(int32_t);
            mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            mf_buffer_init_view(buf, mem, bytes);
            t->buffer = buf;
            t->byte_offset = 0;
            
            int32_t* data = (int32_t*)mem;
            for(int i=0; i<count; ++i) {
                const mf_json_value* item = &val->as.array.items[i];
                if (item->type == MF_JSON_VAL_STRING) data[i] = (int32_t)mf_fnv1a_hash(item->as.s);
                else data[i] = 0;
            }
        }
    }
}

static bool parse_node_attributes(mf_ir_node* dst, const mf_json_value* data, const char* base_path, mf_arena* arena, mf_compiler_diag* diag) {
    if (!data) return true;

    switch (dst->type) {
        case MF_NODE_INPUT:
        case MF_NODE_OUTPUT: {
            const mf_json_value* v_shape = mf_json_get_field(data, "shape");
            const mf_json_value* v_dtype = mf_json_get_field(data, "dtype");
            
            if (dst->type == MF_NODE_INPUT && (!v_shape || v_shape->type != MF_JSON_VAL_ARRAY)) {
                mf_compiler_diag_report(diag, dst->loc, "Input node '%s': missing or invalid 'shape'", dst->id);
                return false;
            }
            
            dst->constant.info.dtype = v_dtype ? mf_dtype_from_str(v_dtype->as.s) : MF_DTYPE_F32;
            
            if (v_shape && v_shape->type == MF_JSON_VAL_ARRAY) {
                dst->constant.info.ndim = (uint8_t)v_shape->as.array.count;
                if (dst->constant.info.ndim > MF_MAX_DIMS) dst->constant.info.ndim = MF_MAX_DIMS;
                for(int k=0; k<dst->constant.info.ndim; ++k) {
                     const mf_json_value* dim = &v_shape->as.array.items[k];
                     if (dim->type == MF_JSON_VAL_NUMBER) {
                         int d = (int)dim->as.n;
                         dst->constant.info.shape[k] = (d < 0) ? -1 : d;
                     }
                }
                mf_shape_calc_strides(&dst->constant.info);
                // For Output nodes, we also copy this to out_shape initially
                if (dst->type == MF_NODE_OUTPUT) {
                    dst->out_shape = dst->constant;
                }
            }
            break;
        }
        case MF_NODE_CONST: {
            const mf_json_value* v_val = mf_json_get_field(data, "value");
            if (v_val) parse_const_tensor(v_val, data, &dst->constant, arena);
            break;
        }
        case MF_NODE_INDEX: {
            const mf_json_value* v_axis = mf_json_get_field(data, "axis");
            if (v_axis && v_axis->type == MF_JSON_VAL_NUMBER) {
                dst->constant.info.dtype = MF_DTYPE_I32;
                dst->constant.info.ndim = 0;
                size_t bytes = sizeof(int32_t);
                mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
                void* mem = MF_ARENA_PUSH(arena, u8, bytes);
                mf_buffer_init_view(buf, mem, bytes);
                dst->constant.buffer = buf;
                *((int32_t*)mem) = (int32_t)v_axis->as.n;
            }
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
    if (!ast) return false;

    out_ir->node_count = ast->node_count;
    out_ir->node_cap = ast->node_count;
    out_ir->nodes = MF_ARENA_PUSH(arena, mf_ir_node, ast->node_count);

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

    // 2. Process Links
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
        if (src_node->type == MF_NODE_CALL) {
            l_dst->src_port_name = l_src->src_port ? mf_arena_strdup(arena, l_src->src_port) : "default";
        } else {
            l_dst->src_port = get_port_index(src_node->type, l_src->src_port);
        }

        // Dest
        if (!mf_map_get(&map, l_src->dst, &l_dst->dst_node_idx)) {
            mf_source_loc loc = {base_path, l_src->loc.line, l_src->loc.column};
            mf_compiler_diag_report(diag, loc, "Link dst '%s' not found", l_src->dst);
            return false;
        }
        mf_ir_node* dst_node = &out_ir->nodes[l_dst->dst_node_idx];
        if (dst_node->type == MF_NODE_CALL) {
            l_dst->dst_port_name = l_src->dst_port ? mf_arena_strdup(arena, l_src->dst_port) : "default";
        } else {
            l_dst->dst_port = get_port_index(dst_node->type, l_src->dst_port);
        }
    }

    return true;
}