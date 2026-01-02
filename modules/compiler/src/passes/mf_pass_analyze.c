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

static const char* _rule_name(mf_out_rule rule) {
    switch(rule) {
        case MF_OUT_SAME_AS_INPUT: return "SAME_AS_INPUT";
        case MF_OUT_SAME_AS_INPUT_2: return "SAME_AS_INPUT_2";
        case MF_OUT_FORCE_F32: return "FORCE_F32";
        case MF_OUT_FORCE_U8: return "FORCE_U8";
        case MF_OUT_FORCE_I32: return "FORCE_I32";
        default: return "Unknown";
    }
}

// --- Helpers ---

static bool check_broadcast(mf_ir_node* node, const mf_tensor* a, const mf_tensor* b, mf_tensor* out, mf_compiler_diag* diag) {
    if (mf_shape_broadcast(&a->info, &b->info, &out->info)) {
        return true;
    }

    char s_a[64], s_b[64];
    mf_shape_format(&a->info, s_a, sizeof(s_a));
    mf_shape_format(&b->info, s_b, sizeof(s_b));
    MF_REPORT(node, "Incompatible shapes for broadcast: %s vs %s", s_a, s_b);
    return false;
}

static mf_identity infer_node_identity(mf_ir_node* node, mf_ir_node* s1, mf_ir_node* s2, mf_ir_node* s3, const mf_op_metadata* meta) {
    if (node->type == MF_NODE_CONST) return MF_IDENTITY_CONSTANT;
    if (node->type == MF_NODE_INDEX) return MF_IDENTITY_SPATIAL;
    if (node->type == MF_NODE_INPUT) return MF_IDENTITY_UNIFORM;
    if (meta->category == MF_OP_CAT_REDUCTION && meta->shape_rule == MF_SHAPE_SCALAR) return MF_IDENTITY_UNIFORM;

    mf_identity id = MF_IDENTITY_CONSTANT;
    mf_ir_node* srcs[] = { s1, s2, s3 };
    for (int i = 0; i < 3; ++i) {
        if (srcs[i]) {
            mf_identity src_id = srcs[i]->out_shape.info.identity;
            if (src_id > id) id = src_id;
        }
    }
    return id;
}

bool mf_pass_analyze(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, mf_compiler_diag* diag) {
    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        if (node->type == MF_NODE_UNKNOWN || node->type >= MF_NODE_COUNT) continue;

        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        
        u32 node_idx = (u32)(node - ir->nodes);
        mf_ir_node* s1 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[0]);
        mf_ir_node* s2 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[1]);
        mf_ir_node* s3 = mf_ir_find_input_by_name(ir, node_idx, meta->ports[2]);

        mf_tensor* out = &node->out_shape;
        
        // Identity is useful for shape rules
        out->info.identity = infer_node_identity(node, s1, s2, s3, meta);

        // --- 1. Resolve Output Shape First ---
        switch (meta->shape_rule) {
            case MF_SHAPE_SPECIAL:
                if (node->type == MF_NODE_CONST) {
                    node->out_shape = node->constant;
                } else if (node->type == MF_NODE_INPUT) {
                    if (!s1) {
                        out->info.ndim = node->constant.info.ndim;
                        memcpy(out->info.shape, node->constant.info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                    } else {
                        out->info.ndim = s1->out_shape.info.ndim;
                        memcpy(out->info.shape, s1->out_shape.info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                    }
                }
                break;

            case MF_SHAPE_SAME_AS_S1:
                if (s1) {
                    out->info.ndim = s1->out_shape.info.ndim;
                    memcpy(out->info.shape, s1->out_shape.info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                } else if (node->type == MF_NODE_OUTPUT) {
                     MF_REPORT(node, "Output '%s' not connected", node->id); return false;
                }
                break;

            case MF_SHAPE_SAME_AS_S2:
                if (s2) {
                    out->info.ndim = s2->out_shape.info.ndim;
                    memcpy(out->info.shape, s2->out_shape.info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                }
                break;

            case MF_SHAPE_BROADCAST:
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs for broadcast op %s", meta->name); return false; }
                if (s3) {
                    mf_tensor tmp;
                    if (!check_broadcast(node, &s1->out_shape, &s2->out_shape, &tmp, diag)) return false;
                    if (!check_broadcast(node, &tmp, &s3->out_shape, out, diag)) return false;
                } else {
                    if (!check_broadcast(node, &s1->out_shape, &s2->out_shape, out, diag)) return false;
                }
                break;

            case MF_SHAPE_MATMUL:
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs for MatMul"); return false; }
                out->info.ndim = 2;
                out->info.shape[0] = s1->out_shape.info.shape[0];
                out->info.shape[1] = s2->out_shape.info.shape[1];
                break;

            case MF_SHAPE_TRANSPOSE:
                if (!s1) return false;
                out->info = s1->out_shape.info;
                if (out->info.ndim == 2) {
                    int32_t t = out->info.shape[0];
                    out->info.shape[0] = out->info.shape[1];
                    out->info.shape[1] = t;
                }
                break;

            case MF_SHAPE_DOT:
                if (!s1) return false;
                out->info.ndim = s1->out_shape.info.ndim > 0 ? s1->out_shape.info.ndim - 1 : 0;
                for(int k=0; k<out->info.ndim; ++k) out->info.shape[k] = s1->out_shape.info.shape[k];
                break;

            case MF_SHAPE_JOIN:
                if (!s1 || !s2) return false;
                out->info = s1->out_shape.info;
                {
                    int components = 2;
                    if (s3) components++;
                    if (mf_ir_find_input_by_name(ir, (u32)(node - ir->nodes), meta->ports[3])) components++;
                    out->info.shape[out->info.ndim++] = components;
                }
                break;

            case MF_SHAPE_GATHER:
                if (!s1 || !s2) return false;
                out->info.ndim = s2->out_shape.info.ndim;
                memcpy(out->info.shape, s2->out_shape.info.shape, sizeof(int32_t) * MF_MAX_DIMS);
                break;

            case MF_SHAPE_RESHAPE:
                if (!s1 || !s2) return false;
                if (mf_tensor_is_valid(&s2->constant)) {
                    int ndim = (int)mf_tensor_count(&s2->constant);
                    out->info.ndim = (uint8_t)ndim;
                    void* d = mf_tensor_data(&s2->constant);
                    for(int k=0; k<ndim && k<MF_MAX_DIMS; ++k) {
                        out->info.shape[k] = (s2->constant.info.dtype == MF_DTYPE_F32) ? (int)((f32*)d)[k] : ((int*)d)[k];
                    }
                }
                break;

            case MF_SHAPE_SLICE:
                if (!s1 || !s2) return false;
                out->info.ndim = 1;
                if (mf_tensor_is_valid(&s2->constant)) {
                    void* d = mf_tensor_data(&s2->constant);
                    out->info.shape[0] = (s2->constant.info.dtype == MF_DTYPE_F32) ? (int)((f32*)d)[1] : ((int*)d)[1];
                }
                break;

            case MF_SHAPE_DYNAMIC_1D:
                out->info.ndim = 1;
                if (node->type == MF_NODE_RANGE && s1 && mf_tensor_is_valid(&s1->constant)) {
                    void* d = mf_tensor_data(&s1->constant);
                    out->info.shape[0] = (s1->constant.info.dtype == MF_DTYPE_F32) ? (int)*((f32*)d) : *((int32_t*)d);
                } else {
                    out->info.shape[0] = 0; // Dynamic
                }
                break;

            case MF_SHAPE_SCALAR:
                out->info.ndim = 0;
                out->info.shape[0] = 1;
                break;
        }

        // --- 2. Resolve Output DType (Must happen AFTER shape resolution) ---
        mf_dtype dtype = out->info.dtype;

        // FORCE rules always take precedence
        if (meta->out_rule == MF_OUT_FORCE_F32) dtype = MF_DTYPE_F32;
        else if (meta->out_rule == MF_OUT_FORCE_U8)  dtype = MF_DTYPE_U8;
        else if (meta->out_rule == MF_OUT_FORCE_I32) dtype = MF_DTYPE_I32;
        // If no force rule and dtype is still unknown, apply propagation rules
        else if (dtype == MF_DTYPE_UNKNOWN) {
            if (meta->out_rule == MF_OUT_SAME_AS_INPUT) {
                if (s1) dtype = s1->out_shape.info.dtype;
                else if (node->constant.info.dtype != MF_DTYPE_UNKNOWN) dtype = node->constant.info.dtype;
            } else if (meta->out_rule == MF_OUT_SAME_AS_INPUT_2) {
                if (s2) dtype = s2->out_shape.info.dtype;
            }
        }

        // Fallback to F32 if still unknown
        if (dtype == MF_DTYPE_UNKNOWN) dtype = MF_DTYPE_F32;
        
        out->info.dtype = dtype;

        // Absolute Validation: Does the final type match the operation's allowed types?
        if (!((1 << dtype) & meta->type_mask)) {
            MF_REPORT(node, "Type Error: Operation '%s' does not support %s output (Allowed mask: 0x%x)", 
                meta->name, _dtype_name(dtype), meta->type_mask);
            return false;
        }

        // Validate Input Types
        if (s1 && !((1 << s1->out_shape.info.dtype) & meta->type_mask)) {
            MF_REPORT(node, "Input 1 '%s' (Type: %s) is incompatible with %s (Allowed mask: 0x%x)", 
                s1->id, _dtype_name(s1->out_shape.info.dtype), meta->name, meta->type_mask);
            return false;
        }
        if (s2 && !((1 << s2->out_shape.info.dtype) & meta->type_mask)) {
            MF_REPORT(node, "Input 2 '%s' (Type: %s) is incompatible with %s (Mask: 0x%x)", 
                s2->id, _dtype_name(s2->out_shape.info.dtype), meta->name, meta->type_mask);
            return false;
        }

        // --- 3. Finalize Strides ---
        mf_shape_calc_strides(&out->info);

        char s_shape[64];
        mf_shape_format(&out->info, s_shape, sizeof(s_shape));
        MF_LOG_TRACE("Analyze: Node %u (%s) -> ID: %s, Identity: %d, Rule: %s, Type: %s, Shape: %s", 
            node_idx, meta->name, node->id, out->info.identity, _rule_name(meta->out_rule), _dtype_name(out->info.dtype), s_shape);
        
        if (s1 || s2 || s3) {
            MF_LOG_TRACE("  Inputs: S1:%s (%s), S2:%s (%s), S3:%s (%s)",
                s1 ? s1->id : "NONE", s1 ? _dtype_name(s1->out_shape.info.dtype) : "N/A",
                s2 ? s2->id : "NONE", s2 ? _dtype_name(s2->out_shape.info.dtype) : "N/A",
                s3 ? s3->id : "NONE", s3 ? _dtype_name(s3->out_shape.info.dtype) : "N/A");
        }
    }

    return true;
}
