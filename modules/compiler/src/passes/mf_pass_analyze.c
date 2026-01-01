#include "../mf_passes.h"
#include "../mf_compiler_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/isa/mf_opcodes.h>
#include <stdio.h>
#include <string.h>

#define MF_REPORT(node, msg, ...) \
    mf_compiler_diag_report(diag, (node)->loc, msg, ##__VA_ARGS__)

static bool check_dtype_match(mf_ir_node* node, mf_tensor* a, mf_tensor* b, mf_compiler_diag* diag) {
    if (a->info.dtype != b->info.dtype) {
        MF_REPORT(node, "Type mismatch: %d vs %d", a->info.dtype, b->info.dtype);
        return false;
    }
    return true;
}

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
    if (sz_a == 1) { *out = *b; return true; }
    if (sz_b == 1) { *out = *a; return true; }
    
    // Strict Match
    if (mf_tensor_same_shape(a, b)) { *out = *a; return true; }
    
    // Simple Suffix Broadcasting: [Batch, N] vs [N]
    if (a->info.ndim == b->info.ndim + 1) {
        bool match = true;
        for (int i=0; i<b->info.ndim; ++i) if (a->info.shape[i+1] != b->info.shape[i]) match = false;
        if (match) { *out = *a; return true; }
    }
    if (b->info.ndim == a->info.ndim + 1) {
        bool match = true;
        for (int i=0; i<a->info.ndim; ++i) if (b->info.shape[i+1] != a->info.shape[i]) match = false;
        if (match) { *out = *b; return true; }
    }

    // Dynamic Batch Broadcasting ([0, N] vs [N])
    if (a->info.shape[0] == 0 && a->info.ndim == b->info.ndim + 1) {
        *out = *a; return true;
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
             // Resolve output shape (prefer non-zero)
             *out = *a;
             for (int i=0; i<a->info.ndim; ++i) {
                 if (out->info.shape[i] == 0) out->info.shape[i] = b->info.shape[i];
             }
             return true; 
        }
    }

    // Format shapes for error message
    char s_a[64], s_b[64];
    format_shape(a, s_a, sizeof(s_a));
    format_shape(b, s_b, sizeof(s_b));

    MF_REPORT(node, "Incompatible shapes for broadcast: %s vs %s", s_a, s_b);
    return false;
}

bool mf_pass_analyze(mf_graph_ir* ir, mf_ir_node** sorted_nodes, size_t count, mf_compiler_diag* diag) {
    for (size_t i = 0; i < count; ++i) {
        mf_ir_node* node = sorted_nodes[i];
        
        // Find inputs
        mf_ir_node* s1 = find_input_source(ir, (u32)(node - ir->nodes), 0);
        mf_ir_node* s2 = find_input_source(ir, (u32)(node - ir->nodes), 1);
        mf_ir_node* s3 = find_input_source(ir, (u32)(node - ir->nodes), 2);

        mf_tensor* out = &node->out_shape;
        memset(out, 0, sizeof(mf_tensor));

        // Propagate / Infer
        switch (node->type) {
            case MF_NODE_CONST:
                node->out_shape = node->constant;
                break;

            case MF_NODE_INPUT:
                 if (s1) {
                     node->out_shape = s1->out_shape;
                 } else {
                     node->out_shape = node->constant;
                 }
                 break;

            case MF_NODE_OUTPUT:
                // Already set by Lower pass or handled in CodeGen. 
                // For Output, we just inherit input.
                 if (node->type == MF_NODE_OUTPUT) {
                     if (!s1) { MF_REPORT(node, "Output not connected"); return false; }
                     node->out_shape = s1->out_shape;
                 }
                 break;

            case MF_NODE_ADD: case MF_NODE_SUB: case MF_NODE_MUL: case MF_NODE_DIV:
            case MF_NODE_MIN: case MF_NODE_MAX: case MF_NODE_POW:
            case MF_NODE_ATAN2: case MF_NODE_STEP:
            {
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs"); return false; }
                if (!check_dtype_match(node, &s1->out_shape, &s2->out_shape, diag)) return false;
                if (!check_broadcast(node, &s1->out_shape, &s2->out_shape, out, diag)) return false;
            } break;

            case MF_NODE_MIX: 
            {
                if (!s1 || !s2 || !s3) { MF_REPORT(node, "Missing inputs for Mix"); return false; }
                // Check all match s1's type (or s3 is float?)
                // Usually Mix(a, b, t) -> t can be float even if a,b are vec.
                // For now strict check.
                if (!check_dtype_match(node, &s1->out_shape, &s2->out_shape, diag)) return false;
                
                // Broadcast s1/s2
                mf_tensor tmp;
                if (!check_broadcast(node, &s1->out_shape, &s2->out_shape, &tmp, diag)) return false;
                // Broadcast result/s3
                if (!check_broadcast(node, &tmp, &s3->out_shape, out, diag)) return false;
            } break;
            
            case MF_NODE_CLAMP: 
            {
                if (!s1 || !s2 || !s3) { MF_REPORT(node, "Missing inputs for Clamp"); return false; }
                if (!check_dtype_match(node, &s1->out_shape, &s2->out_shape, diag)) return false;
                mf_tensor tmp;
                if (!check_broadcast(node, &s1->out_shape, &s2->out_shape, &tmp, diag)) return false;
                if (!check_broadcast(node, &tmp, &s3->out_shape, out, diag)) return false;
            } break;

            case MF_NODE_SMOOTHSTEP:
            {
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs"); return false; }
                // s1: x, s2: edges
                *out = s1->out_shape;
            } break;

            case MF_NODE_SIN: case MF_NODE_COS: case MF_NODE_ABS: case MF_NODE_SQRT:
            case MF_NODE_FLOOR: case MF_NODE_CEIL: case MF_NODE_NOT: 
            case MF_NODE_INVERSE: case MF_NODE_CUMSUM:
            {
                if (!s1) { MF_REPORT(node, "Missing input"); return false; }
                *out = s1->out_shape;
            } break;
            
            case MF_NODE_LENGTH:
            {
                if (!s1) { MF_REPORT(node, "Missing input"); return false; }
                mf_tensor* a = &s1->out_shape;
                out->info.dtype = MF_DTYPE_F32;
                if (a->info.ndim <= 1) {
                     out->info.ndim = 0; // Scalar
                } else {
                     out->info.ndim = a->info.ndim - 1;
                     for(int k=0; k<out->info.ndim; ++k) out->info.shape[k] = a->info.shape[k];
                }
            } break;

            case MF_NODE_LESS: case MF_NODE_GREATER: case MF_NODE_EQUAL:
            case MF_NODE_AND: case MF_NODE_OR:
            {
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs"); return false; }
                // Compare can happen between different types? Usually not.
                if (node->type != MF_NODE_AND && node->type != MF_NODE_OR) {
                     if (!check_dtype_match(node, &s1->out_shape, &s2->out_shape, diag)) return false;
                }
                
                if (!check_broadcast(node, &s1->out_shape, &s2->out_shape, out, diag)) return false;
                out->info.dtype = MF_DTYPE_U8;
            } break;
            
            case MF_NODE_SELECT:
            {
                if (!s2) { MF_REPORT(node, "Missing inputs"); return false; } // s1 (cond) is optional? No, mandated.
                if (!s1) { MF_REPORT(node, "Missing condition"); return false; }
                
                mf_tensor* t = &s2->out_shape;
                mf_tensor* f = s3 ? &s3->out_shape : NULL;
                
                if (f) {
                    if (!check_dtype_match(node, t, f, diag)) return false;
                    // Broadcast T/F
                    mf_tensor tmp;
                    if (!check_broadcast(node, t, f, &tmp, diag)) return false;
                    // Broadcast Result/Cond
                    if (!check_broadcast(node, &tmp, &s1->out_shape, out, diag)) return false;
                } else {
                    // Filter mode? Or just Where(True)? Assuming standard select.
                     if (!check_broadcast(node, t, &s1->out_shape, out, diag)) return false;
                }
                out->info.dtype = t->info.dtype;
            } break;

            case MF_NODE_MATMUL:
            {
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs for MatMul"); return false; }
                mf_tensor* a = &s1->out_shape;
                mf_tensor* b = &s2->out_shape;
                
                if (a->info.ndim != 2 || b->info.ndim != 2) {
                    MF_REPORT(node, "MatMul requires 2D matrices, got %dD and %dD", a->info.ndim, b->info.ndim);
                    return false;
                }

                if (a->info.shape[1] != b->info.shape[0]) {
                    MF_REPORT(node, "MatMul mismatch: [%d,%d] x [%d,%d]", 
                        a->info.shape[0], a->info.shape[1], 
                        b->info.shape[0], b->info.shape[1]);
                    return false;
                }
                
                out->info.dtype = a->info.dtype;
                out->info.ndim = 2;
                out->info.shape[0] = a->info.shape[0];
                out->info.shape[1] = b->info.shape[1];
            } break;

            case MF_NODE_TRANSPOSE:
            {
                if (!s1) { MF_REPORT(node, "Missing input"); return false; }
                *out = s1->out_shape;
                if (out->info.ndim == 2) {
                    int32_t t = out->info.shape[0];
                    out->info.shape[0] = out->info.shape[1];
                    out->info.shape[1] = t;
                }
            } break;

            case MF_NODE_RANGE:
            case MF_NODE_INDEX:
                out->info.dtype = MF_DTYPE_F32; 
                out->info.ndim = 1;
                out->info.shape[0] = 0; // Dynamic
                break;
                
            case MF_NODE_SLICE:
                 if (!s1 || !s2) { MF_REPORT(node, "Missing inputs"); return false; }
                 out->info.dtype = s1->out_shape.info.dtype;
                 out->info.ndim = 1; 
                 // Infer if constant range
                 if (mf_tensor_is_valid(&s2->constant)) {
                      // ... logic ...
                      void* d = mf_tensor_data(&s2->constant);
                      if (s2->constant.info.dtype == MF_DTYPE_F32) out->info.shape[0] = (int)((f32*)d)[1];
                      else out->info.shape[0] = ((int*)d)[1];
                 } else {
                     out->info.shape[0] = 0;
                 }
                 break;

            case MF_NODE_RESHAPE:
                 if (!s1 || !s2) { MF_REPORT(node, "Missing inputs"); return false; }
                 out->info.dtype = s1->out_shape.info.dtype;
                 // Infer if constant shape
                 if (mf_tensor_is_valid(&s2->constant)) {
                     int ndim = (int)mf_tensor_count(&s2->constant);
                     out->info.ndim = (uint8_t)ndim;
                     void* d = mf_tensor_data(&s2->constant);
                     for(int k=0; k<ndim && k<MF_MAX_DIMS; ++k) {
                         if (s2->constant.info.dtype == MF_DTYPE_F32) out->info.shape[k] = (int)((f32*)d)[k];
                         else out->info.shape[k] = ((int*)d)[k];
                     }
                 }
                 break;
            
            case MF_NODE_GATHER:
                if (!s1 || !s2) { MF_REPORT(node, "Missing inputs for Gather (Data, Indices)"); return false; }
                // Output Type follows Data
                out->info.dtype = s1->out_shape.info.dtype;
                // Output Shape follows Indices
                out->info.ndim = s2->out_shape.info.ndim;
                for(int k=0; k<out->info.ndim; ++k) out->info.shape[k] = s2->out_shape.info.shape[k];
                break;
                 
            case MF_NODE_JOIN:
                 if (!s1 || !s2) { MF_REPORT(node, "Missing inputs"); return false; }
                 if (!check_dtype_match(node, &s1->out_shape, &s2->out_shape, diag)) return false;
                 // Check shapes equal or scalar
                 // ... simplified for now
                 *out = s1->out_shape;
                 out->info.shape[out->info.ndim++] = 2;
                 break;
                 
            case MF_NODE_DOT:
                 if (!s1 || !s2) { MF_REPORT(node, "Missing inputs"); return false; }
                 out->info.dtype = MF_DTYPE_F32;
                 out->info.ndim = s1->out_shape.info.ndim > 0 ? s1->out_shape.info.ndim - 1 : 0;
                 // Copy dims...
                 for(int k=0; k<out->info.ndim; ++k) out->info.shape[k] = s1->out_shape.info.shape[k];
                 break;

            default: break;
        }
    }
    return true;
}
