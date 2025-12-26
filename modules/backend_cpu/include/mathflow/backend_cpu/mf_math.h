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

// --- Matrix 4x4 ---

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

static inline mf_mat4 mf_mat4_transpose(mf_mat4 m) {
    mf_mat4 res;
    for(int c=0; c<4; ++c) {
        for(int r=0; r<4; ++r) {
            res.m[c*4+r] = m.m[r*4+c];
        }
    }
    return res;
}

// Minimal implementation of 4x4 inverse using cofactor expansion
static inline mf_mat4 mf_mat4_inverse(mf_mat4 m) {
    f32 inv[16];
    f32 det;
    f32* mat = m.m;

    inv[0] = mat[5]  * mat[10] * mat[15] - 
             mat[5]  * mat[11] * mat[14] - 
             mat[9]  * mat[6]  * mat[15] + 
             mat[9]  * mat[7]  * mat[14] +
             mat[13] * mat[6]  * mat[11] - 
             mat[13] * mat[7]  * mat[10];

    inv[4] = -mat[4]  * mat[10] * mat[15] + 
              mat[4]  * mat[11] * mat[14] + 
              mat[8]  * mat[6]  * mat[15] - 
              mat[8]  * mat[7]  * mat[14] - 
              mat[12] * mat[6]  * mat[11] + 
              mat[12] * mat[7]  * mat[10];

    inv[8] = mat[4]  * mat[9] * mat[15] - 
             mat[4]  * mat[11] * mat[13] - 
             mat[8]  * mat[5] * mat[15] + 
             mat[8]  * mat[7] * mat[13] + 
             mat[12] * mat[5] * mat[11] - 
             mat[12] * mat[7] * mat[9];

    inv[12] = -mat[4]  * mat[9] * mat[14] + 
               mat[4]  * mat[10] * mat[13] +
               mat[8]  * mat[5] * mat[14] - 
               mat[8]  * mat[6] * mat[13] - 
               mat[12] * mat[5] * mat[10] + 
               mat[12] * mat[6] * mat[9];

    inv[1] = -mat[1]  * mat[10] * mat[15] + 
              mat[1]  * mat[11] * mat[14] + 
              mat[9]  * mat[2] * mat[15] - 
              mat[9]  * mat[3] * mat[14] - 
              mat[13] * mat[2] * mat[11] + 
              mat[13] * mat[3] * mat[10];

    inv[5] = mat[0]  * mat[10] * mat[15] - 
             mat[0]  * mat[11] * mat[14] - 
             mat[8]  * mat[2] * mat[15] + 
             mat[8]  * mat[3] * mat[14] + 
             mat[12] * mat[2] * mat[11] - 
             mat[12] * mat[3] * mat[10];

    inv[9] = -mat[0]  * mat[9] * mat[15] + 
              mat[0]  * mat[11] * mat[13] + 
              mat[8]  * mat[1] * mat[15] - 
              mat[8]  * mat[3] * mat[13] - 
              mat[12] * mat[1] * mat[11] + 
              mat[12] * mat[3] * mat[9];

    inv[13] = mat[0]  * mat[9] * mat[14] - 
              mat[0]  * mat[10] * mat[13] - 
              mat[8]  * mat[1] * mat[14] + 
              mat[8]  * mat[2] * mat[13] + 
              mat[12] * mat[1] * mat[10] - 
              mat[12] * mat[2] * mat[9];

    inv[2] = mat[1]  * mat[6] * mat[15] - 
             mat[1]  * mat[7] * mat[14] - 
             mat[5]  * mat[2] * mat[15] + 
             mat[5]  * mat[3] * mat[14] + 
             mat[13] * mat[2] * mat[7] - 
             mat[13] * mat[3] * mat[6];

    inv[6] = -mat[0]  * mat[6] * mat[15] + 
              mat[0]  * mat[7] * mat[14] + 
              mat[4]  * mat[2] * mat[15] - 
              mat[4]  * mat[3] * mat[14] - 
              mat[12] * mat[2] * mat[7] + 
              mat[12] * mat[3] * mat[6];

    inv[10] = mat[0]  * mat[5] * mat[15] - 
              mat[0]  * mat[7] * mat[13] - 
              mat[4]  * mat[1] * mat[15] + 
              mat[4]  * mat[3] * mat[13] + 
              mat[12] * mat[1] * mat[7] - 
              mat[12] * mat[3] * mat[5];

    inv[14] = -mat[0]  * mat[5] * mat[14] + 
               mat[0]  * mat[6] * mat[13] + 
               mat[4]  * mat[1] * mat[14] - 
               mat[4]  * mat[2] * mat[13] - 
               mat[12] * mat[1] * mat[6] + 
               mat[12] * mat[2] * mat[5];

    inv[3] = -mat[1] * mat[6] * mat[11] + 
              mat[1] * mat[7] * mat[10] + 
              mat[5] * mat[2] * mat[11] - 
              mat[5] * mat[3] * mat[10] - 
              mat[9] * mat[2] * mat[7] + 
              mat[9] * mat[3] * mat[6];

    inv[7] = mat[0] * mat[6] * mat[11] - 
             mat[0] * mat[7] * mat[10] - 
             mat[4] * mat[2] * mat[11] + 
             mat[4] * mat[3] * mat[10] + 
             mat[8] * mat[2] * mat[7] - 
             mat[8] * mat[3] * mat[6];

    inv[11] = -mat[0] * mat[5] * mat[11] + 
               mat[0] * mat[7] * mat[9] + 
               mat[4] * mat[1] * mat[11] - 
               mat[4] * mat[3] * mat[9] - 
               mat[8] * mat[1] * mat[7] + 
               mat[8] * mat[3] * mat[5];

    inv[15] = mat[0] * mat[5] * mat[10] - 
              mat[0] * mat[6] * mat[9] - 
              mat[4] * mat[1] * mat[10] + 
              mat[4] * mat[2] * mat[9] + 
              mat[8] * mat[1] * mat[6] - 
              mat[8] * mat[2] * mat[5];

    det = mat[0] * inv[0] + mat[1] * inv[4] + mat[2] * inv[8] + mat[3] * inv[12];

    if (det == 0) return mf_mat4_identity();

    det = 1.0 / det;

    mf_mat4 res;
    for (int i = 0; i < 16; i++) res.m[i] = inv[i] * det;
    return res;
}

// --- Matrix 3x3 ---

static inline mf_mat3 mf_mat3_identity(void) {
    mf_mat3 res = {0};
    res.m[0] = 1.0f; res.m[4] = 1.0f; res.m[8] = 1.0f;
    return res;
}

static inline mf_mat3 mf_mat3_mul(mf_mat3 a, mf_mat3 b) {
    mf_mat3 res;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            res.m[i * 3 + j] = 0;
            for (int k = 0; k < 3; k++) {
                res.m[i * 3 + j] += a.m[i * 3 + k] * b.m[k * 3 + j];
            }
        }
    }
    return res;
}

static inline mf_mat3 mf_mat3_transpose(mf_mat3 m) {
    mf_mat3 res;
    for(int c=0; c<3; ++c) {
        for(int r=0; r<3; ++r) {
            res.m[c*3+r] = m.m[r*3+c];
        }
    }
    return res;
}

static inline f32 mf_mat3_det(mf_mat3 m) {
    return m.m[0] * (m.m[4] * m.m[8] - m.m[5] * m.m[7]) -
           m.m[3] * (m.m[1] * m.m[8] - m.m[2] * m.m[7]) +
           m.m[6] * (m.m[1] * m.m[5] - m.m[2] * m.m[4]);
}

static inline mf_mat3 mf_mat3_inverse(mf_mat3 m) {
    f32 det = mf_mat3_det(m);
    if (fabsf(det) < 1e-6f) return mf_mat3_identity(); // Fallback
    f32 invDet = 1.0f / det;
    
    mf_mat3 res;
    // Adjugate matrix / det (Transposed Cofactors)
    // 00 -> +(44*88 - 55*77)
    res.m[0] = (m.m[4] * m.m[8] - m.m[5] * m.m[7]) * invDet;
    // 10 -> -(10*88 - 20*70) -> 01 of original
    res.m[1] = (m.m[2] * m.m[7] - m.m[1] * m.m[8]) * invDet;
    res.m[2] = (m.m[1] * m.m[5] - m.m[2] * m.m[4]) * invDet;
    
    res.m[3] = (m.m[5] * m.m[6] - m.m[3] * m.m[8]) * invDet;
    res.m[4] = (m.m[0] * m.m[8] - m.m[2] * m.m[6]) * invDet;
    res.m[5] = (m.m[2] * m.m[3] - m.m[0] * m.m[5]) * invDet;
    
    res.m[6] = (m.m[3] * m.m[7] - m.m[4] * m.m[6]) * invDet;
    res.m[7] = (m.m[1] * m.m[6] - m.m[0] * m.m[7]) * invDet;
    res.m[8] = (m.m[0] * m.m[4] - m.m[1] * m.m[3]) * invDet;
    return res;
}

#endif // MF_MATH_H