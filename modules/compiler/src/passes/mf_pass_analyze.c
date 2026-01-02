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

// --- Helpers ---

static bool check_broadcast(mf_ir_node* node, const mf_tensor* a, const mf_tensor* b, mf_tensor* out, mf_compiler_diag* diag) {
    if (mf_shape_broadcast(&a->info, &b->info, &out->info)) {
        return true;
    }

    char s_a[64], s_b[64];
    mf_shape_format(&a->info, s_b, sizeof(s_b));
    MF_REPORT(node, "Incompatible shapes for broadcast: %s vs %s", s_a, s_b);
    return false;
}

static mf_identity infer_node_identity(mf_ir_node* node, mf_ir_node* s1, mf_ir_node* s2, mf_ir_node* s3, const mf_op_metadata* meta) {
    if (node->type == MF_NODE_CONST) return MF_IDENTITY_CONSTANT;
    if (node->type == MF_NODE_INDEX) return MF_IDENTITY_SPATIAL;
    
    // Inputs are usually Uniform (u_Time, etc.) unless they are bound to spatial resources.
    // In our current host model, system inputs like Time/Mouse are UNIFORM.
    if (node->type == MF_NODE_INPUT) return MF_IDENTITY_UNIFORM;

    // Reductions (like Sum) effectively collapse spatial data into a single value.
    if (meta->category == MF_OP_CAT_REDUCTION && meta->shape_rule == MF_SHAPE_SCALAR) {
        return MF_IDENTITY_UNIFORM;
    }

    // Default propagation: result is as complex as the most complex input.
    // Identity order: CONSTANT (1) < UNIFORM (2) < SPATIAL (3)
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
        mf_ir_node* s1 = find_input_source(ir, node_idx, 0);
        mf_ir_node* s2 = find_input_source(ir, node_idx, 1);
        mf_ir_node* s3 = find_input_source(ir, node_idx, 2);

        mf_tensor* out = &node->out_shape;
        
        // --- 0. Identity Inference ---
        out->info.identity = infer_node_identity(node, s1, s2, s3, meta);

        // 1. Validate Types
        mf_dtype existing_dtype = out->info.dtype;
        if (out->info.ndim == 0 && out->info.shape[0] == 0) {
            memset(out, 0, sizeof(mf_tensor));
        }
        if (existing_dtype != MF_DTYPE_UNKNOWN) out->info.dtype = existing_dtype;

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
                    // If Output has a pre-defined shape (from JSON), and s1 is dynamic, 
                    // we keep the Output shape.
                    bool out_has_shape = (node->out_shape.info.ndim > 0 || node->out_shape.info.shape[0] > 0);
                    if (node->type == MF_NODE_OUTPUT && out_has_shape) {
                        // Keep pre-defined shape, but check compatibility if s1 also has shape
                        if (s1->out_shape.info.shape[0] > 0 && s1->out_shape.info.shape[0] != node->out_shape.info.shape[0]) {
                             // Potential warning/error if not broadcastable
                        }
                        out->info = node->out_shape.info;
                    } else {
                        out->info = s1->out_shape.info;
                    }
                } else if (node->type == MF_NODE_OUTPUT) {
                     MF_REPORT(node, "Output not connected"); return false;
                }
                break;

            case MF_SHAPE_SAME_AS_S2:
                if (s2) out->info = s2->out_shape.info;
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
                    if (find_input_source(ir, (u32)(node - ir->nodes), 3)) components++;
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
                } else if (node->type == MF_NODE_INDEX) {
                    out->info.shape[0] = 0; // Truly dynamic based on execution context
                } else {
                    out->info.shape[0] = 0; // Dynamic
                }
                break;

            case MF_SHAPE_SCALAR:
                out->info.ndim = 0;
                out->info.shape[0] = 1;
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

                // If identity is UNIFORM or CONSTANT, it should NOT step with the domain.

                // Its "effective" stride for the execution model is 0.

                // Note: internal strides of the tensor itself (e.g. for a 3x3 matrix uniform) 

                // are still needed, but the compiler's STEP_N strides in codegen will use this info.

                

                for (int k = out->info.ndim - 1; k >= 0; --k) {

                    out->info.strides[k] = stride;

                    stride *= (out->info.shape[k] > 0 ? out->info.shape[k] : 1);

                }

        

                MF_LOG_TRACE("Analyze: Node %u (%s) -> Identity: %d, Shape: [%d]", 

                    node_idx, meta->name, out->info.identity, out->info.shape[0]);

            }

            return true;

        }

        