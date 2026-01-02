#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_op_defs.h>
#include <stdio.h>
#include <string.h>

#define MF_REPORT(node, msg, ...) \
    mf_compiler_diag_report(diag, (node)->loc, msg, ##__VA_ARGS__)

static const char* _dtype_name(mf_dtype type) {
    switch(type) {
        case MF_DTYPE_F32: return "F32";
        case MF_DTYPE_I32: return "I32";
        case MF_DTYPE_U8:  return "U8";
        case MF_DTYPE_UNKNOWN: return "Unknown";
        default: return "Invalid";
    }
}

// --- Helpers ---

static bool check_broadcast(mf_ir_node* node, const mf_type_info* a, const mf_type_info* b, mf_type_info* out, mf_compiler_diag* diag) {
    if (mf_shape_broadcast(a, b, out)) {
        return true;
    }

    char s_a[64], s_b[64];
    mf_shape_format(a, s_a, sizeof(s_a));
    mf_shape_format(b, s_b, sizeof(s_b));
    MF_REPORT(node, "Incompatible shapes for broadcast: %s vs %s", s_a, s_b);
    return false;
}

static const mf_compile_port* find_port(const mf_compile_contract* contract, const char* name, bool is_input) {
    if (!contract || !name) return NULL;
    uint32_t count = is_input ? contract->input_count : contract->output_count;
    mf_compile_port* ports = is_input ? contract->inputs : contract->outputs;
    for (uint32_t i = 0; i < count; ++i) {
        if (ports[i].name && strcmp(ports[i].name, name) == 0) return &ports[i];
    }
    return NULL;
}

bool mf_pass_analyze(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, const mf_compile_contract* contract, mf_compiler_diag* diag) {
    // --- 0. Pre-pass: Initialize shapes from contract ---
    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        if (node->type == MF_NODE_INPUT || node->type == MF_NODE_OUTPUT) {
            const mf_compile_port* cp = find_port(contract, node->id, (node->type == MF_NODE_INPUT));
            if (cp) {
                node->out_info.dtype = cp->dtype;
                node->out_info.ndim = cp->ndim;
                memcpy(node->out_info.shape, cp->shape, sizeof(int32_t) * MF_MAX_DIMS);
                mf_shape_calc_strides(&node->out_info);
                
                node->builtin_id = cp->builtin_id;
                node->builtin_axis = cp->builtin_axis;
            }
        }
    }

    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        if (node->type == MF_NODE_UNKNOWN || node->type >= MF_NODE_COUNT) continue;

        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        u32 node_idx = (u32)(node - ir->nodes);
        
        mf_ir_node* s1 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[0]);
        mf_ir_node* s2 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[1]);
        mf_ir_node* s3 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[2]);
        mf_ir_node* s4 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[3]);

        mf_type_info* out = &node->out_info;
        char s_before[64] = "unknown";
        mf_shape_format(out, s_before, sizeof(s_before));

        // --- 1. Resolve Output Shape ---
        switch (meta->shape_rule) {
            case MF_SHAPE_SPECIAL:
                if (node->type == MF_NODE_CONST) {
                    node->out_info = node->const_info;
                } else if (node->type == MF_NODE_INPUT || node->type == MF_NODE_OUTPUT) {
                    if (node->type == MF_NODE_INPUT) {
                        if (node->builtin_id == MF_BUILTIN_INDEX) {
                            u32 dom_node_idx = node->domain_node_idx;
                            if (out->ndim == 0) {
                                if (dom_node_idx == UINT32_MAX) {
                                    for (u32 j = 0; j < (u32)ir->node_count; ++j) {
                                        if (ir->nodes[j].type == MF_NODE_OUTPUT) { dom_node_idx = j; break; }
                                    }
                                }
                                if (dom_node_idx != UINT32_MAX && dom_node_idx != node_idx) {
                                    *out = ir->nodes[dom_node_idx].out_info;
                                }
                            }
                            if (out->dtype == MF_DTYPE_UNKNOWN) out->dtype = MF_DTYPE_F32;
                        } else if (s1 && out->ndim == 0) {
                            *out = s1->out_info;
                        } else if (out->ndim == 0) {
                            *out = node->const_info;
                        }
                    } else if (node->type == MF_NODE_OUTPUT) {
                        if (out->ndim == 0 && s1) *out = s1->out_info;
                        if (node->domain_node_idx == UINT32_MAX && s1) {
                            node->domain_node_idx = (s1->domain_node_idx == UINT32_MAX) ? (u32)(s1 - ir->nodes) : s1->domain_node_idx;
                        }
                    }
                }
                break;

            case MF_SHAPE_SAME_AS_S1:
                if (s1) {
                    out->ndim = s1->out_info.ndim;
                    memcpy(out->shape, s1->out_info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                } else if (node->type == MF_NODE_OUTPUT) {
                     MF_REPORT(node, "Output '%s' not connected", node->id); return false;
                }
                break;

            case MF_SHAPE_SAME_AS_S2:
                if (s2) {
                    out->ndim = s2->out_info.ndim;
                    memcpy(out->shape, s2->out_info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                }
                break;

            case MF_SHAPE_BROADCAST:
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs for broadcast op %s", meta->name); return false; }
                if (s3) {
                    mf_type_info tmp;
                    if (!check_broadcast(node, &s1->out_info, &s2->out_info, &tmp, diag)) return false;
                    if (!check_broadcast(node, &tmp, &s3->out_info, out, diag)) return false;
                } else {
                    if (!check_broadcast(node, &s1->out_info, &s2->out_info, out, diag)) return false;
                }
                break;

            case MF_SHAPE_MATMUL:
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs for MatMul"); return false; }
                out->ndim = 2;
                out->shape[0] = s1->out_info.shape[s1->out_info.ndim - 2];
                out->shape[1] = s2->out_info.shape[s2->out_info.ndim - 1];
                break;

            case MF_SHAPE_TRANSPOSE:
                if (!s1) return false;
                *out = s1->out_info;
                if (out->ndim >= 2) {
                    int32_t t = out->shape[out->ndim - 2];
                    out->shape[out->ndim - 2] = out->shape[out->ndim - 1];
                    out->shape[out->ndim - 1] = t;
                }
                break;

            case MF_SHAPE_DOT:
                if (!s1) return false;
                out->ndim = s1->out_info.ndim > 0 ? s1->out_info.ndim - 1 : 0;
                for(int k=0; k<out->ndim; ++k) out->shape[k] = s1->out_info.shape[k];
                break;

            case MF_SHAPE_JOIN:
                if (!s1 || !s2) return false;
                *out = s1->out_info;
                {
                    int components = 2;
                    if (s3) components++;
                    if (mf_ir_find_input_by_name(ir, (u32)(node - ir->nodes), meta->ports[3])) components++;
                    out->shape[out->ndim++] = components;
                }
                break;

            case MF_SHAPE_GATHER:
                if (!s1 || !s2) return false;
                out->ndim = s2->out_info.ndim;
                memcpy(out->shape, s2->out_info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                break;

            case MF_SHAPE_RESHAPE:
                if (!s1 || !s2) return false;
                if (s2->const_data) {
                    int count = (int)mf_shape_calc_count(s2->const_info.shape, s2->const_info.ndim);
                    out->ndim = (uint8_t)count;
                    void* d = s2->const_data;
                    for(int k=0; k<count && k<MF_MAX_DIMS; ++k) {
                        out->shape[k] = (s2->const_info.dtype == MF_DTYPE_F32) ? (int)((f32*)d)[k] : ((int*)d)[k];
                    }
                }
                break;

            case MF_SHAPE_SLICE:
                if (!s1 || !s2) return false;
                out->ndim = 1;
                if (s2->const_data) {
                    void* d = s2->const_data;
                    out->shape[0] = (s2->const_info.dtype == MF_DTYPE_F32) ? (int)((f32*)d)[1] : ((int*)d)[1];
                }
                break;

            case MF_SHAPE_SCALAR:
                out->ndim = 0;
                out->shape[0] = 1;
                break;
        }

        // --- 2. Resolve Output DType ---
        mf_dtype dtype = MF_DTYPE_UNKNOWN;
        if (meta->out_rule == MF_OUT_FORCE_F32) dtype = MF_DTYPE_F32;
        else if (meta->out_rule == MF_OUT_FORCE_U8)  dtype = MF_DTYPE_U8;
        else if (meta->out_rule == MF_OUT_FORCE_I32) dtype = MF_DTYPE_I32;
        else {
            if (meta->out_rule == MF_OUT_SAME_AS_INPUT) {
                if (s1) dtype = s1->out_info.dtype;
                else if (node->const_info.dtype != MF_DTYPE_UNKNOWN) dtype = node->const_info.dtype;
            } else if (meta->out_rule == MF_OUT_SAME_AS_INPUT_2) {
                if (s2) dtype = s2->out_info.dtype;
            }
        }
        if (dtype == MF_DTYPE_UNKNOWN) dtype = (out->dtype != MF_DTYPE_UNKNOWN) ? out->dtype : MF_DTYPE_F32;
        out->dtype = dtype;

        // Validation
        if (!((1 << dtype) & meta->output_mask)) {
            MF_REPORT(node, "Type Error: Operation '%s' does not support %s output", meta->name, _dtype_name(dtype));
            return false;
        }
        if (s1 && !((1 << s1->out_info.dtype) & meta->input_mask)) {
            MF_REPORT(node, "Input 1 '%s' (%s) incompatible with %s", s1->id, _dtype_name(s1->out_info.dtype), meta->name);
            return false;
        }
        if (s2 && !((1 << s2->out_info.dtype) & meta->input_mask)) {
            MF_REPORT(node, "Input 2 '%s' (%s) incompatible with %s", s2->id, _dtype_name(s2->out_info.dtype), meta->name);
            return false;
        }

        // --- 3. Finalize Strides ---
        mf_shape_calc_strides(out);
        u32 actual_dom_idx = (node->domain_node_idx == UINT32_MAX) ? node_idx : node->domain_node_idx;
        size_t task_dom_count = mf_shape_calc_count(ir->nodes[actual_dom_idx].out_info.shape, ir->nodes[actual_dom_idx].out_info.ndim);
        node->is_spatial = (task_dom_count > 1);

        size_t dom_count = mf_shape_calc_count(out->shape, out->ndim);
        node->strides[0] = node->is_spatial ? (i32)mf_shape_calc_linear_stride(dom_count, task_dom_count) : 0;
        node->strides[1] = s1 ? (i32)mf_shape_calc_linear_stride(mf_shape_calc_count(s1->out_info.shape, s1->out_info.ndim), task_dom_count) : 0;
        node->strides[2] = s2 ? (i32)mf_shape_calc_linear_stride(mf_shape_calc_count(s2->out_info.shape, s2->out_info.ndim), task_dom_count) : 0;
        node->strides[3] = s3 ? (i32)mf_shape_calc_linear_stride(mf_shape_calc_count(s3->out_info.shape, s3->out_info.ndim), task_dom_count) : 0;

        char s_shape[64];
        mf_shape_format(out, s_shape, sizeof(s_shape));
        MF_LOG_INFO("Analyze: Node %u (%s) ID:%s -> Shape:%s (Before:%s), Type:%s, Strides:[%d,%d,%d,%d]", 
            node_idx, meta->name, node->id, s_shape, s_before, _dtype_name(out->dtype),
            node->strides[0], node->strides[1], node->strides[2], node->strides[3]);
    }
    return true;
}