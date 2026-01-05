#include <mathflow/base/mf_shape.h>
#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <string.h>

void mf_shape_calc_strides(mf_type_info* info) {
    int32_t stride = 1;
    for (int k = (int)info->ndim - 1; k >= 0; --k) {
        info->strides[k] = stride;
        stride *= (info->shape[k] > 0 ? info->shape[k] : 1);
    }
}

size_t mf_shape_calc_count(const int32_t* shape, uint8_t ndim) {
    if (ndim == 0) return 1;
    size_t count = 1;
    for (int i = 0; i < ndim; ++i) {
        count *= (shape[i] > 0 ? (size_t)shape[i] : 1);
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
    
    info->ndim = new_ndim;
    if (new_ndim > 0) {
        memcpy(info->shape, new_shape, sizeof(int32_t) * new_ndim);
    }
    mf_shape_calc_strides(info);
}

void mf_shape_infer_strides(const mf_type_info* shape, const mf_type_info* domain, int32_t* out_strides) {
    for (int i = 0; i < MF_MAX_DIMS; ++i) out_strides[i] = 0;
    if (shape->ndim == 0) return;
    
    int32_t current_stride = 1;
    int s_idx = (int)shape->ndim - 1;
    int d_idx = (int)domain->ndim - 1;
    
    while (s_idx >= 0 && d_idx >= 0) {
        if (shape->shape[s_idx] == domain->shape[d_idx]) {
            out_strides[d_idx] = current_stride;
            current_stride *= shape->shape[s_idx];
        } else if (shape->shape[s_idx] == 1) {
            out_strides[d_idx] = 0;
        } else {
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
        offset += snprintf(buf + offset, size - offset, "%d%s", info->shape[i], i < (int)info->ndim - 1 ? "," : "");
    }
    if (offset < (int)size) snprintf(buf + offset, size - offset, "]");
}

bool mf_shape_broadcast(const mf_type_info* a, const mf_type_info* b, mf_type_info* out) {
    if (mf_shape_is_scalar(a)) { *out = *b; return true; }
    if (mf_shape_is_scalar(b)) { *out = *a; return true; }
    
    int ndim_a = (int)a->ndim;
    int ndim_b = (int)b->ndim;
    int max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    
    out->ndim = (uint8_t)max_ndim;
    out->dtype = a->dtype;
    
    for (int i = 0; i < max_ndim; ++i) {
        int idx_a = ndim_a - 1 - i;
        int idx_b = ndim_b - 1 - i;
        int idx_out = max_ndim - 1 - i;
        
        int32_t dim_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int32_t dim_b = (idx_b >= 0) ? b->shape[idx_b] : 1;
        
        if (dim_a == dim_b) out->shape[idx_out] = dim_a;
        else if (dim_a == 1) out->shape[idx_out] = dim_b;
        else if (dim_b == 1) out->shape[idx_out] = dim_a;
        else if (dim_a < 0 || dim_b < 0) out->shape[idx_out] = (dim_a > 0) ? dim_a : dim_b;
        else return false;
    }
    
    mf_shape_calc_strides(out);
    return true;
}

i32 mf_shape_calc_linear_stride(size_t op_count, size_t dom_count) {
    if (dom_count <= 1) return (op_count > 0) ? 1 : 0;
    if (op_count == dom_count) return 1;
    if (op_count == 1) return 0;
    if (op_count > dom_count && (op_count % dom_count) == 0) return (i32)(op_count / dom_count);
    return 0;
}