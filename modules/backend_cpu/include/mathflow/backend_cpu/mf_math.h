#ifndef MF_MATH_H
#define MF_MATH_H

#include <math.h>
#include <string.h>
#include <mathflow/isa/mf_base.h>

// --- Operations ---

static inline mf_vec2 mf_vec2_add(mf_vec2 a, mf_vec2 b) {
    return (mf_vec2){a.x + b.x, a.y + b.y};
}

static inline mf_vec3 mf_vec3_add(mf_vec3 a, mf_vec3 b) {
    return (mf_vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline f32 mf_vec3_dot(mf_vec3 a, mf_vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline mf_vec3 mf_vec3_cross(mf_vec3 a, mf_vec3 b) {
    return (mf_vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline mf_vec3 mf_vec3_normalize(mf_vec3 v) {
    f32 len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0) {
        f32 inv_len = 1.0f / len;
        return (mf_vec3){v.x * inv_len, v.y * inv_len, v.z * inv_len};
    }
    return (mf_vec3){0, 0, 0};
}

static inline mf_mat4 mf_mat4_identity(void) {
    mf_mat4 res = {0};
    res.m[0] = 1.0f; res.m[5] = 1.0f; res.m[10] = 1.0f; res.m[15] = 1.0f;
    return res;
}

static inline mf_mat4 mf_mat4_translate(mf_vec3 v) {
    mf_mat4 res = mf_mat4_identity();
    res.m[12] = v.x;
    res.m[13] = v.y;
    res.m[14] = v.z;
    return res;
}

static inline mf_mat4 mf_mat4_mul(mf_mat4 a, mf_mat4 b) {
    mf_mat4 res;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            res.m[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                res.m[i * 4 + j] += a.m[i * 4 + k] * b.m[k * 4 + j];
            }
        }
    }
    return res;
}

#endif // MF_MATH_H