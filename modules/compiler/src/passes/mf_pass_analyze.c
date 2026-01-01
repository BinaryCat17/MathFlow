#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_op_defs.h>
#include <stdio.h>
#include <string.h>

#define MF_REPORT(node, msg, ...) \
    mf_compiler_diag_report(diag, (node)->loc, msg, ##__VA_ARGS__)

// --- Helpers ---

static void format_shape(const mf_tensor* t, char* buf, size_t size) {
    if (t->info.ndim == 0) {
        snprintf(buf, size, "[]");
        return;
    }
    int offset = snprintf(buf, size, "[");
    for (int i = 0; i < t->info.ndim; ++i) {
        if (offset >= (int)size) break;
        offset += snprintf(buf + offset, size - offset, "%d%s", t->info.shape[i], i < t->info.ndim - 1 ? "," : "");
    }
    if (offset < (int)size) snprintf(buf + offset, size - offset, "]");
}

static bool check_broadcast(mf_ir_node* node, const mf_tensor* a, const mf_tensor* b, mf_tensor* out, mf_compiler_diag* diag) {
    size_t sz_a = mf_tensor_count(a);
    size_t sz_b = mf_tensor_count(b);
    
    // Scalar Broadcast
    if (sz_a == 1) { out->info = b->info; return true; }
    if (sz_b == 1) { out->info = a->info; return true; }
    
    // Strict Match
    if (mf_tensor_same_shape(a, b)) { out->info = a->info; return true; }
    
    // Simple Suffix Broadcasting: [Batch, N] vs [N]
    if (a->info.ndim == b->info.ndim + 1) {
        bool match = true;
        for (int i=0; i<b->info.ndim; ++i) if (a->info.shape[i+1] != b->info.shape[i]) match = false;
        if (match) { out->info = a->info; return true; }
    }
    if (b->info.ndim == a->info.ndim + 1) {
        bool match = true;
        for (int i=0; i<a->info.ndim; ++i) if (b->info.shape[i+1] != a->info.shape[i]) match = false;
        if (match) { out->info = b->info; return true; }
    }

    // Dynamic Batch Broadcasting ([0, N] vs [N])
    if (a->info.shape[0] == 0 && a->info.ndim == b->info.ndim + 1) {
        out->info = a->info; return true;
    }
    
    // Dynamic Wildcard Match (Treat 0 as 'Any')
    if (a->info.ndim == b->info.ndim) {
        bool match = true;
        for (int i=0; i<a->info.ndim; ++i) {
             if (a->info.shape[i] != b->info.shape[i]) {
                 if (a->info.shape[i] != 0 && b->info.shape[i] != 0) match = false;
             }
        }
        if (match) { 
             out->info = a->info;
             for (int i=0; i<a->info.ndim; ++i) {
                 if (out->info.shape[i] == 0) out->info.shape[i] = b->info.shape[i];
             }
             return true; 
        }
    }

    char s_a[64], s_b[64];
    format_shape(a, s_a, sizeof(s_a));
    format_shape(b, s_b, sizeof(s_b));
    MF_REPORT(node, "Incompatible shapes for broadcast: %s vs %s", s_a, s_b);
    return false;
}

bool mf_pass_analyze(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, mf_compiler_diag* diag) {
    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        if (node->type == MF_NODE_UNKNOWN || node->type >= MF_NODE_COUNT) continue;

        const mf_op_metadata* meta = &MF_OP_METADATA[node->type];
        
        mf_ir_node* s1 = find_input_source(ir, (u32)(node - ir->nodes), 0);
        mf_ir_node* s2 = find_input_source(ir, (u32)(node - ir->nodes), 1);
        mf_ir_node* s3 = find_input_source(ir, (u32)(node - ir->nodes), 2);

        mf_tensor* out = &node->out_shape;
        memset(out, 0, sizeof(mf_tensor));

        // 1. Validate Types
        if (s1 && !((1 << s1->out_shape.info.dtype) & meta->type_mask)) {
            MF_REPORT(node, "Input 1 has invalid type %d for operation", s1->out_shape.info.dtype);
            return false;
        }
        if (s2 && !((1 << s2->out_shape.info.dtype) & meta->type_mask)) {
            MF_REPORT(node, "Input 2 has invalid type %d for operation", s2->out_shape.info.dtype);
            return false;
        }

        // 3. Resolve Output Shape (Rule based)
        switch (meta->shape_rule) {
            case MF_SHAPE_SPECIAL:
                if (node->type == MF_NODE_CONST) {
                    node->out_shape = node->constant;
                } else if (node->type == MF_NODE_INPUT) {
                    node->out_shape = s1 ? s1->out_shape : node->constant;
                }
                break;

            case MF_SHAPE_SAME_AS_S1:
                if (s1) {
                    out->info = s1->out_shape.info;
                } else if (node->type == MF_NODE_OUTPUT) {
                     MF_REPORT(node, "Output not connected"); return false;
                }
                break;

            case MF_SHAPE_BROADCAST:
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs for broadcast op"); return false; }
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
                if (s1->out_shape.info.ndim != 2 || s2->out_shape.info.ndim != 2) {
                    MF_REPORT(node, "MatMul requires 2D matrices"); return false;
                }
                if (s1->out_shape.info.shape[1] != s2->out_shape.info.shape[0]) {
                    MF_REPORT(node, "MatMul dimension mismatch"); return false;
                }
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
                if (!s1 || !s2) return false;
                out->info.ndim = s1->out_shape.info.ndim > 0 ? s1->out_shape.info.ndim - 1 : 0;
                for(int k=0; k<out->info.ndim; ++k) out->info.shape[k] = s1->out_shape.info.shape[k];
                break;

            case MF_SHAPE_JOIN:
                if (!s1 || !s2) return false;
                out->info = s1->out_shape.info;
                out->info.shape[out->info.ndim++] = 2;
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
                out->info.shape[0] = 0; // Dynamic
                break;
        }

        // 2. Resolve Output DType (Must happen AFTER shape, as rules might set info)
        switch (meta->out_rule) {
            case MF_OUT_SAME_AS_INPUT:   out->info.dtype = s1 ? s1->out_shape.info.dtype : MF_DTYPE_F32; break;
            case MF_OUT_SAME_AS_INPUT_2: out->info.dtype = s2 ? s2->out_shape.info.dtype : MF_DTYPE_F32; break;
            case MF_OUT_FORCE_F32:       out->info.dtype = MF_DTYPE_F32; break;
            case MF_OUT_FORCE_U8:        out->info.dtype = MF_DTYPE_U8;  break;
            case MF_OUT_FORCE_I32:       out->info.dtype = MF_DTYPE_I32; break;
        }

        // 4. Finalize strides (contiguous)
        int32_t stride = 1;
        for (int k = out->info.ndim - 1; k >= 0; --k) {
            out->info.strides[k] = stride;
            stride *= (out->info.shape[k] > 0 ? out->info.shape[k] : 1);
        }
    }
    return true;
}