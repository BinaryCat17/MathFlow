#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_op_defs.h>
#include <stdio.h>
#include <string.h>

static bool check_broadcast(mf_ir_node* node, const mf_type_info* a, const mf_type_info* b, mf_type_info* out, mf_compiler_diag* diag) {
    if (mf_shape_broadcast(a, b, out)) return true;
    char s_a[64], s_b[64];
    mf_shape_format(a, s_a, sizeof(s_a));
    mf_shape_format(b, s_b, sizeof(s_b));
    MF_REPORT_NODE(diag, node, "Incompatible shapes for broadcast: %s vs %s", s_a, s_b);
    return false;
}

bool mf_pass_analyze(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, mf_compiler_diag* diag) {
    // 0. Pre-pass: Port & Constant Initialization
    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        
        if (node->type == MF_NODE_INPUT || node->type == MF_NODE_OUTPUT) {
            // Autonomous mode: use info already in node (from JSON or defaults)
            node->out_info = node->const_info;
            mf_shape_normalize(&node->out_info);
        }

        if (node->type == MF_NODE_CONST) {
            mf_shape_normalize(&node->const_info);
            node->out_info = node->const_info;
        }
    }

    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        if (node->type == MF_NODE_UNKNOWN || node->type >= MF_NODE_COUNT) continue;

        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        u32 node_idx = (u32)(node - ir->nodes);
        
        mf_ir_node* inputs[4] = {0};
        for (u8 k = 0; k < 4; ++k) {
            if (meta->ports[k]) inputs[k] = mf_ir_find_input_by_name(ir, node_idx, meta->ports[k]);
        }

        mf_type_info* out = &node->out_info;

        // 1. Resolve Output Shape
        switch (meta->shape_rule) {
            case MF_SHAPE_SPECIAL:
                if (node->type == MF_NODE_CONST) { /* Handled in pre-pass */ }
                else if (node->type == MF_NODE_INPUT) {
                    if (node->builtin_id == MF_BUILTIN_INDEX) {
                        u32 dom_idx = node->domain_node_idx;
                        if (dom_idx == UINT32_MAX) {
                            for (u32 j = 0; j < (u32)ir->node_count; ++j) if (ir->nodes[j].type == MF_NODE_OUTPUT) { dom_idx = j; break; }
                        }
                        if (dom_idx != UINT32_MAX && dom_idx != node_idx) *out = ir->nodes[dom_idx].out_info;
                        if (out->dtype == MF_DTYPE_UNKNOWN) out->dtype = MF_DTYPE_F32;
                    } else if (inputs[0] && out->ndim == 0) *out = inputs[0]->out_info;
                    else if (out->ndim == 0) *out = node->const_info;
                } else if (node->type == MF_NODE_OUTPUT) {
                    if (out->ndim == 0 && inputs[0]) *out = inputs[0]->out_info;
                    if (node->domain_node_idx == UINT32_MAX && inputs[0])
                        node->domain_node_idx = (inputs[0]->domain_node_idx == UINT32_MAX) ? (u32)(inputs[0] - ir->nodes) : inputs[0]->domain_node_idx;
                }
                break;

            case MF_SHAPE_SAME_AS_S1: if (inputs[0]) { out->ndim = inputs[0]->out_info.ndim; memcpy(out->shape, inputs[0]->out_info.shape, sizeof(int32_t)*MF_MAX_DIMS); } else { MF_REPORT_NODE(diag, node, "Missing S1 input for %s", meta->name); return false; } break;
            case MF_SHAPE_SAME_AS_S2: if (inputs[1]) { out->ndim = inputs[1]->out_info.ndim; memcpy(out->shape, inputs[1]->out_info.shape, sizeof(int32_t)*MF_MAX_DIMS); } else { MF_REPORT_NODE(diag, node, "Missing S2 input for %s", meta->name); return false; } break;
            case MF_SHAPE_BROADCAST:
                if (!inputs[0] || !inputs[1]) { MF_REPORT_NODE(diag, node, "Missing inputs for broadcast in %s", meta->name); return false; }
                if (inputs[2]) { mf_type_info tmp; if (!check_broadcast(node, &inputs[0]->out_info, &inputs[1]->out_info, &tmp, diag)) return false; if (!check_broadcast(node, &tmp, &inputs[2]->out_info, out, diag)) return false; }
                else if (!check_broadcast(node, &inputs[0]->out_info, &inputs[1]->out_info, out, diag)) return false;
                break;

            case MF_SHAPE_MATMUL:
                if (!inputs[0] || !inputs[1]) { MF_REPORT_NODE(diag, node, "Missing inputs for matmul"); return false; }
                out->ndim = 2;
                out->shape[0] = inputs[0]->out_info.shape[inputs[0]->out_info.ndim - 2];
                out->shape[1] = inputs[1]->out_info.shape[inputs[1]->out_info.ndim - 1];
                break;

            case MF_SHAPE_TRANSPOSE: if (!inputs[0]) { MF_REPORT_NODE(diag, node, "Missing input for transpose"); return false; } *out = inputs[0]->out_info; if (out->ndim >= 2) { int32_t t = out->shape[out->ndim-2]; out->shape[out->ndim-2] = out->shape[out->ndim-1]; out->shape[out->ndim-1] = t; } break;
            case MF_SHAPE_DOT: if (!inputs[0]) { MF_REPORT_NODE(diag, node, "Missing input for dot"); return false; } out->ndim = inputs[0]->out_info.ndim > 0 ? inputs[0]->out_info.ndim - 1 : 0; for(int k=0; k<out->ndim; ++k) out->shape[k] = inputs[0]->out_info.shape[k]; break;
            case MF_SHAPE_JOIN: if (!inputs[0] || !inputs[1]) { MF_REPORT_NODE(diag, node, "Missing inputs for join"); return false; } *out = inputs[0]->out_info; { int comps = 2; if (inputs[2]) comps++; if (inputs[3]) comps++; out->shape[out->ndim++] = comps; } break;
            case MF_SHAPE_GATHER: if (!inputs[1]) { MF_REPORT_NODE(diag, node, "Missing indices for gather"); return false; } out->ndim = inputs[1]->out_info.ndim; memcpy(out->shape, inputs[1]->out_info.shape, sizeof(int32_t)*MF_MAX_DIMS); break;
            case MF_SHAPE_RESHAPE: if (!inputs[1] || !inputs[1]->const_data) { MF_REPORT_NODE(diag, node, "Reshape needs constant shape input"); return false; } { int cnt = (int)mf_shape_calc_count(inputs[1]->const_info.shape, inputs[1]->const_info.ndim); out->ndim = (uint8_t)cnt; for(int k=0; k<cnt && k<MF_MAX_DIMS; ++k) out->shape[k] = (inputs[1]->const_info.dtype == MF_DTYPE_F32) ? (int)((f32*)inputs[1]->const_data)[k] : ((int*)inputs[1]->const_data)[k]; } break;
            case MF_SHAPE_SLICE: if (!inputs[1] || !inputs[1]->const_data) { MF_REPORT_NODE(diag, node, "Slice needs constant range input"); return false; } out->ndim = 1; out->shape[0] = (inputs[1]->const_info.dtype == MF_DTYPE_F32) ? (int)((f32*)inputs[1]->const_data)[1] : ((int*)inputs[1]->const_data)[1]; break;
            case MF_SHAPE_SCALAR: out->ndim = 0; out->shape[0] = 1; break;
        }

        // 2. Resolve DType
        mf_dtype dtype = MF_DTYPE_UNKNOWN;
        if (meta->out_rule == MF_OUT_FORCE_F32) dtype = MF_DTYPE_F32;
        else if (meta->out_rule == MF_OUT_FORCE_U8) dtype = MF_DTYPE_U8;
        else if (meta->out_rule == MF_OUT_FORCE_I32) dtype = MF_DTYPE_I32;
        else if (meta->out_rule == MF_OUT_SAME_AS_INPUT && inputs[0]) dtype = inputs[0]->out_info.dtype;
        else if (meta->out_rule == MF_OUT_SAME_AS_INPUT_2 && inputs[1]) dtype = inputs[1]->out_info.dtype;
        
        if (dtype == MF_DTYPE_UNKNOWN) dtype = (out->dtype != MF_DTYPE_UNKNOWN) ? out->dtype : MF_DTYPE_F32;
        out->dtype = dtype;

        // 3. Strides & Spatial Analysis
        mf_shape_calc_strides(out);
        u32 dom_idx = (node->domain_node_idx == UINT32_MAX) ? node_idx : node->domain_node_idx;
        size_t task_cnt = mf_shape_calc_count(ir->nodes[dom_idx].out_info.shape, ir->nodes[dom_idx].out_info.ndim);
        
        bool is_generator = (node->builtin_id != MF_BUILTIN_NONE);
        bool has_spatial_input = false;
        for (int k = 0; k < 4; ++k) if (inputs[k] && inputs[k]->is_spatial) has_spatial_input = true;

        node->is_spatial = (task_cnt > 1) || is_generator || has_spatial_input;

        // Inflation for generators: they must match the domain to produce a stream
        if (is_generator && task_cnt > 1) {
            const mf_type_info* dom_info = &ir->nodes[dom_idx].out_info;
            out->ndim = dom_info->ndim;
            memcpy(out->shape, dom_info->shape, sizeof(int32_t) * dom_info->ndim);
            mf_shape_calc_strides(out);
        }

        node->strides[0] = (i32)mf_shape_calc_linear_stride(mf_shape_calc_count(out->shape, out->ndim), task_cnt);
        for (int k = 0; k < 4; ++k) {
            if (inputs[k]) {
                node->strides[k+1] = (i32)mf_shape_calc_linear_stride(mf_shape_calc_count(inputs[k]->out_info.shape, inputs[k]->out_info.ndim), task_cnt);
            } else {
                node->strides[k+1] = 0;
            }
        }
    }
    return true;
}
