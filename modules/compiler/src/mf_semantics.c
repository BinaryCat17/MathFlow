#include "mf_compiler_internal.h"
#include <mathflow/isa/mf_opcodes.h>
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

// --- Shape Inference Logic ---

bool mf_infer_shape(mf_ir_node* node, mf_ir_node* s1, mf_ir_node* s2, mf_ir_node* s3) {
    mf_tensor* out = &node->out_shape;
    memset(out, 0, sizeof(mf_tensor));

    switch (node->type) {
        case MF_NODE_ADD: case MF_NODE_SUB: case MF_NODE_MUL: case MF_NODE_DIV:
        case MF_NODE_MIN: case MF_NODE_MAX: case MF_NODE_ATAN2: case MF_NODE_POW:
        case MF_NODE_CLAMP:
        {
            if (s1 && s2) {
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                
                // Validation: Both must be same shape or one must be scalar
                bool a_scal = (a->size == 1);
                bool b_scal = (b->size == 1);
                
                if (!a_scal && !b_scal && !mf_tensor_same_shape(a, b)) {
                    printf("Error: Shape mismatch in node '%s'. Input shapes: ", node->id);
                    printf("[%d] vs [%d]\n", a->shape[0], b->shape[0]);
                    return false;
                }

                *out = a_scal ? *b : *a;
            }
        } break;

        case MF_NODE_SIN: case MF_NODE_COS: case MF_NODE_ABS: case MF_NODE_SQRT:
        case MF_NODE_FLOOR: case MF_NODE_CEIL: case MF_NODE_NOT:
        {
            // Unary: Copy shape from first input
            if (s1) *out = s1->out_shape;
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
                        printf("Error: MatMul shape mismatch in node '%s'. Inner dimensions [%d] and [%d] do not match.\n", node->id, a->shape[1], b->shape[0]);
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
                        printf("Error: MatMul shape mismatch in node '%s'. Sizes %zu and %zu do not match.\n", node->id, a->size, b->size);
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
                         printf("Error: Select shape mismatch in node '%s'.\n", node->id);
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
            // Output is 1D Array of F32. Size is dynamic (determined by input value).
            out->dtype = MF_DTYPE_F32; 
            out->ndim = 1;
            out->shape[0] = 0; // Dynamic
            out->size = 0;
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

        case MF_NODE_MEMORY:
            // Memory node output shape is initially defined by its 'init' value.
            // If an input is connected, it may trigger a resize during execution
            // if the input shape differs from the initial state.
            *out = node->constant;
            break;

        default: break;
    }
    return true;
}
