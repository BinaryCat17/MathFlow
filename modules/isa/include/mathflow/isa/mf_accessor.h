#ifndef MF_ACCESSOR_H
#define MF_ACCESSOR_H

#include <mathflow/isa/mf_tensor_iter.h>

/**
 * Typed Accessors for MathFlow Tensors.
 * They wrap mf_tensor_iter and provide type-safe get/set methods
 * with optional bounds and type checking.
 */

#define MF_DEFINE_ACCESSOR(TYPE, SUFFIX, DTYPE_ENUM) \
typedef struct { \
    mf_tensor_iter it; \
} mf_accessor_##SUFFIX; \
\
static inline mf_accessor_##SUFFIX mf_accessor_##SUFFIX##_begin(const mf_tensor* t) { \
    mf_accessor_##SUFFIX acc; \
    acc.it = mf_tensor_iter_begin(t); \
    /* Strict Type Check in Debug */ \
    assert(t->info.dtype == DTYPE_ENUM || t->info.dtype == MF_DTYPE_UNKNOWN); \
    return acc; \
} \
\
static inline TYPE mf_accessor_##SUFFIX##_get(const mf_accessor_##SUFFIX* acc) { \
    return *((TYPE*)acc->it.ptr); \
} \
\
static inline void mf_accessor_##SUFFIX##_set(mf_accessor_##SUFFIX* acc, TYPE val) { \
    *((TYPE*)acc->it.ptr) = val; \
} \
\
static inline void mf_accessor_##SUFFIX##_advance(mf_accessor_##SUFFIX* acc, i32 step) { \
    mf_tensor_iter_advance(&acc->it, step); \
}

MF_DEFINE_ACCESSOR(f32, f32, MF_DTYPE_F32)
MF_DEFINE_ACCESSOR(int32_t, i32, MF_DTYPE_I32)
MF_DEFINE_ACCESSOR(u8, u8, MF_DTYPE_U8)

#undef MF_DEFINE_ACCESSOR

/**
 * Special Safe Accessor for F32 that automatically handles NaN/Inf.
 */
static inline f32 mf_accessor_f32_get_safe(const mf_accessor_f32* acc) {
    f32 val = *((f32*)acc->it.ptr);
    return (isfinite(val)) ? val : 0.0f;
}

#endif // MF_ACCESSOR_H
