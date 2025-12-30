#include "mf_compiler_internal.h"
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
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

// --- Helper: Shape Compatibility ---
static bool mf_shapes_compatible(const mf_tensor* a, const mf_tensor* b, mf_tensor* out) {
    bool a_scal = (a->size == 1);
    bool b_scal = (b->size == 1);
    
    if (a_scal) { *out = *b; return true; }
    if (b_scal) { *out = *a; return true; }
    
    // Strict match
    if (mf_tensor_same_shape(a, b)) { *out = *a; return true; }
    
    // Simple Broadcasting: [Batch, N] vs [N]
    if (a->ndim == b->ndim + 1) {
        // Check suffix
        bool match = true;
        for (int i=0; i<b->ndim; ++i) if (a->shape[i+1] != b->shape[i]) match = false;
        if (match) { *out = *a; return true; }
    }
    if (b->ndim == a->ndim + 1) {
        bool match = true;
        for (int i=0; i<a->ndim; ++i) if (b->shape[i+1] != a->shape[i]) match = false;
        if (match) { *out = *b; return true; }
    }
    
    // Dynamic Batch Broadcasting ([0, N] vs [N])
    // If shape[0] == 0, treat as Wildcard
    if (a->shape[0] == 0 && a->ndim == b->ndim + 1) {
        // Assume match
        *out = *a; return true;
    }

    return false;
}

// --- Shape Inference Logic ---

bool mf_infer_shape(mf_ir_node* node, mf_ir_node* s1, mf_ir_node* s2, mf_ir_node* s3) {
    mf_tensor* out = &node->out_shape;
    memset(out, 0, sizeof(mf_tensor));

    switch (node->type) {
        case MF_NODE_ADD: case MF_NODE_SUB: case MF_NODE_MUL: case MF_NODE_DIV:
        case MF_NODE_MIN: case MF_NODE_MAX: case MF_NODE_ATAN2: case MF_NODE_POW:
        case MF_NODE_STEP:
        {
            if (s1 && s2) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                
                if (!mf_shapes_compatible(a, b, out)) {
                    MF_LOG_ERROR("Shape mismatch in node '%s'. Input shapes: [%d] vs [%d]", node->id, a->ndim, b->ndim);
                    return false;
                }
            }
        } break;

        case MF_NODE_MIX: case MF_NODE_CLAMP: 
        {
             if (s1 && s2 && s3) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                mf_tensor* c = &s3->out_shape;
                
                // Find the non-scalar shape (if any)
                mf_tensor* shape = a;
                if (a->size == 1) shape = b;
                if (shape->size == 1) shape = c;
                
                // Verify all non-scalars match 'shape'
                if (a->size > 1 && !mf_tensor_same_shape(a, shape)) return false;
                if (b->size > 1 && !mf_tensor_same_shape(b, shape)) return false;
                if (c->size > 1 && !mf_tensor_same_shape(c, shape)) return false;

                *out = *shape;
             }
        } break;
        
        case MF_NODE_SMOOTHSTEP:
        {
            if (s1 && s2) {
                // s1: Value, s2: Edges [..., 2]
                *out = s1->out_shape;
                // We rely on runtime checks for s2 shape compatibility
            }
        } break;

        case MF_NODE_JOIN:
        {
            if (s1 && s2) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                // For MVP, require strict match or scalar broadcast
                if (a->size == b->size) {
                    *out = *a;
                } else if (a->size == 1) {
                    *out = *b;
                } else if (b->size == 1) {
                    *out = *a;
                } else {
                     return false;
                }
                
                // Add Dimension
                out->shape[out->ndim] = 2;
                out->ndim += 1;
                out->size *= 2;
            }
        } break;

        case MF_NODE_DOT:
        {
            if (s1 && s2) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                // Assume last dim match
                out->dtype = MF_DTYPE_F32;
                if (a->ndim <= 1) {
                    out->ndim = 0;
                    out->size = 1;
                } else {
                    out->ndim = a->ndim - 1;
                    for(int i=0; i<out->ndim; ++i) out->shape[i] = a->shape[i];
                    out->size = a->size / a->shape[a->ndim-1];
                }
            }
        } break;

        case MF_NODE_SIN: case MF_NODE_COS: case MF_NODE_ABS: case MF_NODE_SQRT:
        case MF_NODE_FLOOR: case MF_NODE_CEIL: case MF_NODE_NOT: case MF_NODE_LENGTH:
        case MF_NODE_EXPORT_INPUT: case MF_NODE_EXPORT_OUTPUT:
        {
            // Unary: Copy shape from first input
            if (s1) {
                 if (node->type == MF_NODE_LENGTH) {
                     mf_tensor* a = &s1->out_shape;
                     out->dtype = MF_DTYPE_F32;
                     if (a->ndim <= 1) {
                         out->ndim = 0;
                         out->size = 1;
                     } else {
                         out->ndim = a->ndim - 1;
                         for(int i=0; i<out->ndim; ++i) out->shape[i] = a->shape[i];
                         out->size = a->size / a->shape[a->ndim-1];
                     }
                 } else {
                     *out = s1->out_shape;
                 }
            }
        } break;

        case MF_NODE_LESS: case MF_NODE_GREATER: case MF_NODE_EQUAL:
        case MF_NODE_AND: case MF_NODE_OR:
        {
            // Comparison/Logic: Shape from larger input, but DType is U8
            if (s1 && s2) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                *out = (a->size >= b->size) ? *a : *b;
                out->dtype = MF_DTYPE_U8;
            }
        } break;

        case MF_NODE_MATMUL:
            if (s1 && s2) {
                // A: [M, K], B: [K, N] -> C: [M, N]
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                if (a->ndim == 2 && b->ndim == 2) {
                    if (a->shape[1] != b->shape[0]) {
                        MF_LOG_ERROR("MatMul shape mismatch in node '%s'. Inner dimensions [%d] and [%d] do not match.", node->id, a->shape[1], b->shape[0]);
                        return false;
                    }
                    out->dtype = a->dtype;
                    out->ndim = 2;
                    out->shape[0] = a->shape[0];
                    out->shape[1] = b->shape[1];
                    out->size = out->shape[0] * out->shape[1];
                } else {
                    // Fallback
                    if (a->size != b->size) {
                        MF_LOG_ERROR("MatMul shape mismatch in node '%s'. Sizes %zu and %zu do not match.", node->id, a->size, b->size);
                        return false;
                    }
                    *out = *a; 
                }
            }
            break;

        case MF_NODE_TRANSPOSE:
            if (s1) {
                *out = s1->out_shape;
                if (out->ndim == 2) {
                    int32_t tmp = out->shape[0];
                    out->shape[0] = out->shape[1];
                    out->shape[1] = tmp;
                }
            }
            break;

        case MF_NODE_SELECT:
            if (s2) {
                mf_tensor* t = &s2->out_shape;
                mf_tensor* f = s3 ? &s3->out_shape : NULL;

                if (f) {
                    bool t_s = (t->size == 1);
                    bool f_s = (f->size == 1);
                    if (!t_s && !f_s && !mf_tensor_same_shape(t, f)) {
                         MF_LOG_ERROR("Select shape mismatch in node '%s'.", node->id);
                         return false;
                    }
                    *out = t_s ? *f : *t;
                } else {
                    *out = *t;
                }
            }
            break;
            
        case MF_NODE_INVERSE:
            if (s1) *out = s1->out_shape;
            break;

        case MF_NODE_RANGE:
        case MF_NODE_INDEX:
        case MF_NODE_RESOLUTION:
            // Output is 1D Array of F32. Size is dynamic (determined by input value or batch context).
            out->dtype = MF_DTYPE_F32; 
            out->ndim = 1;
            out->shape[0] = 0; // Dynamic
            out->size = 0;
            // Correction for Resolution: It is always scalar [1].
            if (node->type == MF_NODE_RESOLUTION) {
                out->ndim = 0;
                out->size = 1;
            }
            break;

        case MF_NODE_CUMSUM:
            if (s1) *out = s1->out_shape;
            break;

        case MF_NODE_COMPRESS:
            if (s1 && s2) {
                // S1: Data, S2: Mask
                *out = s1->out_shape;
                // Shape is dynamic (subset of input)
                out->shape[0] = 0; 
                out->size = 0;
            }
            break;

        default: break;
    }
    return true;
}
