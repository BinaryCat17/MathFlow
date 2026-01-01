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
