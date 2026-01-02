#include <mathflow/base/mf_shape.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <string.h>

void mf_shape_calc_strides(mf_type_info* info) {
    int32_t stride = 1;
    for (int k = info->ndim - 1; k >= 0; --k) {
        info->strides[k] = stride;
        stride *= (info->shape[k] > 0 ? info->shape[k] : 1);
    }
}

void mf_shape_format(const mf_type_info* info, char* buf, size_t size) {
    if (info->ndim == 0) {
        snprintf(buf, size, "[]");
        return;
    }
    int offset = snprintf(buf, size, "[");
    for (int i = 0; i < info->ndim; ++i) {
        if (offset >= (int)size) break;
        offset += snprintf(buf + offset, size - offset, "%d%s", info->shape[i], i < info->ndim - 1 ? "," : "");
    }
    if (offset < (int)size) snprintf(buf + offset, size - offset, "]");
}

bool mf_shape_broadcast(const mf_type_info* a, const mf_type_info* b, mf_type_info* out) {
    size_t sz_a = 1;
    for(int i=0; i<a->ndim; ++i) sz_a *= (a->shape[i] > 0 ? a->shape[i] : 1);
    size_t sz_b = 1;
    for(int i=0; i<b->ndim; ++i) sz_b *= (b->shape[i] > 0 ? b->shape[i] : 1);
    
    // Scalar Broadcast
    if (sz_a == 1 && a->ndim == 0) { *out = *b; return true; }
    if (sz_b == 1 && b->ndim == 0) { *out = *a; return true; }
    
    // Strict Match
    bool same = (a->ndim == b->ndim);
    if (same) {
        for(int i=0; i<a->ndim; ++i) if(a->shape[i] != b->shape[i]) same = false;
    }
    if (same) { *out = *a; return true; }
    
    // Simple Suffix Broadcasting: [Batch, N] vs [N]
    if (a->ndim == b->ndim + 1) {
        bool match = true;
        for (int i=0; i<b->ndim; ++i) if (a->shape[i+1] != b->shape[i]) match = false;
        if (match) { *out = *a; return true; }
    }
    if (b->ndim == a->ndim + 1) {
        bool match = true;
        for (int i=0; i<a->ndim; ++i) if (b->shape[i+1] != a->shape[i]) match = false;
        if (match) { *out = *b; return true; }
    }

    // Dynamic Wildcard Match (Treat 0 or -1 as 'Any')
    if (a->ndim == b->ndim) {
        bool match = true;
        out->ndim = a->ndim;
        out->dtype = a->dtype;
        for (int i=0; i<a->ndim; ++i) {
             if (a->shape[i] != b->shape[i]) {
                 if (a->shape[i] > 0 && b->shape[i] > 0) match = false;
                 else out->shape[i] = (a->shape[i] > 0) ? a->shape[i] : b->shape[i];
             } else {
                 out->shape[i] = a->shape[i];
             }
        }
        if (match) {
            mf_shape_calc_strides(out);
            return true;
        }
    }

    return false;
}

i32 mf_shape_calc_linear_stride(size_t op_count, size_t dom_count) {
    if (dom_count <= 1) return (op_count > 0) ? 1 : 0;
    
    if (op_count == dom_count || op_count == 0) return 1;
    if (op_count == 1) return 0;
    
    if (op_count > dom_count && (op_count % dom_count == 0)) {
        size_t stride = op_count / dom_count;
        if (stride <= 16) return (i32)stride; 
    }
    
    return 0; 
}

bool mf_shape_is_compatible(const mf_type_info* port, const mf_type_info* res, bool is_output) {
    if (port->dtype != res->dtype && port->dtype != MF_DTYPE_UNKNOWN) return false;

    // If port has no shape defined, it's compatible with anything (it will adopt the resource shape)
    if (port->ndim == 0) return true;

    // Output ports must match exactly (for static dimensions)
    // or be a prefix of the resource shape (e.g. [H, W] -> [H, W, 4])
    if (is_output) {
        if (res->ndim < port->ndim) return false;
        for (int i = 0; i < port->ndim; ++i) {
            if (port->shape[i] > 0 && res->shape[i] > 0 && port->shape[i] != res->shape[i]) {
                return false;
            }
        }
        return true;
    }

    // Input ports can accept broadcasted resources
    mf_type_info tmp;
    if (!mf_shape_broadcast(port, res, &tmp)) return false;

    // Result of broadcast must be equal to port (port can be larger or equal to res)
    if (tmp.ndim != port->ndim) return false;
    for (int i = 0; i < tmp.ndim; ++i) {
        if (tmp.shape[i] != port->shape[i]) return false;
    }

    return true;
}
