#include <mathflow/base/mf_shape.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <string.h>

size_t mf_shape_calc_count(const int32_t* shape, uint8_t ndim) {
    if (ndim == 0) return 1;
    size_t count = 1;
    for (int i = 0; i < ndim; ++i) {
        count *= (shape[i] > 0 ? shape[i] : 1);
    }
    return count;
}

size_t mf_shape_calc_bytes(mf_dtype dtype, const int32_t* shape, uint8_t ndim) {
    return mf_shape_calc_count(shape, ndim) * mf_dtype_size(dtype);
}

bool mf_shape_is_scalar(const mf_type_info* info) {
    if (info->ndim == 0) return true;
    for (int i = 0; i < info->ndim; ++i) {
        if (info->shape[i] > 1) return false;
    }
    return true;
}

void mf_shape_normalize(mf_type_info* info) {
    if (info->ndim == 0) return;
    
    int32_t new_shape[MF_MAX_DIMS];
    uint8_t new_ndim = 0;
    
    for (int i = 0; i < info->ndim; ++i) {
        if (info->shape[i] != 1) {
            new_shape[new_ndim++] = info->shape[i];
        }
    }
    
    // If all dimensions were 1, it's a scalar []
    // Special case: we might want to keep at least 1D if it's [1]
    // But for pure normalization, [] is better.
    
    info->ndim = new_ndim;
    if (new_ndim > 0) {
        memcpy(info->shape, new_shape, sizeof(int32_t) * new_ndim);
    }
    mf_shape_calc_strides(info);
}

void mf_shape_calc_strides(mf_type_info* info) {
    int32_t stride = 1;
    for (int k = info->ndim - 1; k >= 0; --k) {
        info->strides[k] = stride;
        stride *= (info->shape[k] > 0 ? info->shape[k] : 1);
    }
}

void mf_shape_infer_strides(const mf_type_info* shape, const mf_type_info* domain, int32_t* out_strides) {
    // Default to 0 (broadcast)
    for (int i = 0; i < MF_MAX_DIMS; ++i) out_strides[i] = 0;
    
    if (shape->ndim == 0) return; // Scalar remains 0
    
    // Match dimensions from tail to head (NumPy style broadcasting)
    int32_t current_stride = 1;
    int s_idx = shape->ndim - 1;
    int d_idx = domain->ndim - 1;
    
    while (s_idx >= 0 && d_idx >= 0) {
        if (shape->shape[s_idx] == domain->shape[d_idx]) {
            out_strides[d_idx] = current_stride;
            current_stride *= shape->shape[s_idx];
        } else if (shape->shape[s_idx] == 1) {
            out_strides[d_idx] = 0;
        } else {
            // Incompatible shapes for linear iteration! 
            // Fallback to 0 or we might need to report error.
            out_strides[d_idx] = 0;
        }
        s_idx--;
        d_idx--;
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
    // 1. Handle Scalars (any 1-element tensor)
    if (mf_shape_is_scalar(a)) { *out = *b; return true; }
    if (mf_shape_is_scalar(b)) { *out = *a; return true; }
    
    // 2. NumPy-style Broadcasting (Align Right)
    int ndim_a = a->ndim;
    int ndim_b = b->ndim;
    int max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    
    out->ndim = (uint8_t)max_ndim;
    out->dtype = a->dtype; // Assume compatible dtypes for now (checked elsewhere)
    
    for (int i = 0; i < max_ndim; ++i) {
        int idx_a = ndim_a - 1 - i;
        int idx_b = ndim_b - 1 - i;
        int idx_out = max_ndim - 1 - i;
        
        int32_t dim_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int32_t dim_b = (idx_b >= 0) ? b->shape[idx_b] : 1;
        
        if (dim_a == dim_b) {
            out->shape[idx_out] = dim_a;
        } else if (dim_a == 1) {
            out->shape[idx_out] = dim_b;
        } else if (dim_b == 1) {
            out->shape[idx_out] = dim_a;
        } else if (dim_a < 0 || dim_b < 0) {
            // Wildcards: take the specific dimension if available
            out->shape[idx_out] = (dim_a > 0) ? dim_a : dim_b;
        } else {
            return false; // Incompatible
        }
    }
    
    mf_shape_calc_strides(out);
    return true;
}

i32 mf_shape_calc_linear_stride(size_t op_count, size_t dom_count) {
    // A scalar (1 element) always has stride 0 to stay fixed during iteration (broadcasting)
    if (op_count == 1) return 0;
    
    if (dom_count <= 1) {
        return (op_count == 0) ? 1 : 0;
    }
    
    if (op_count == 0) return 1;

    // Support for vector streams: op_count is multiple of dom_count
    if (op_count >= dom_count && (op_count % dom_count) == 0) {
        return (i32)(op_count / dom_count);
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
