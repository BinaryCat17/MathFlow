#ifndef MF_TYPES_H
#define MF_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>

// --- Basic Types ---
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;

typedef float    f32;
typedef double   f64;

#define MF_KB(x) ((x) * 1024LL)
#define MF_MB(x) (MF_KB(x) * 1024LL)
#define MF_GB(x) (MF_MB(x) * 1024LL)

// --- Math Types ---
typedef struct { f32 x, y; } mf_vec2;
typedef struct { f32 x, y, z; } mf_vec3;
typedef struct { f32 x, y, z, w; } mf_vec4;

// Column-major 4x4 matrix
typedef struct { f32 m[16]; } mf_mat4;

// Column-major 3x3 matrix
typedef struct { f32 m[9]; } mf_mat3;

#define MF_MAX_DIMS 8

// --- Data Types ---
typedef enum {
    MF_DTYPE_UNKNOWN = 0,
    MF_DTYPE_F32,   // Standard float
    MF_DTYPE_I32,   // Integer / String ID
    MF_DTYPE_U8,    // Byte / Bool
    MF_DTYPE_COUNT
} mf_dtype;

// --- Tensor Metadata (Value Semantics) ---
// Describes the "Shape" of data, independent of storage.
typedef struct {
    mf_dtype dtype;
    uint8_t ndim; // Rank
    int32_t shape[MF_MAX_DIMS];
    int32_t strides[MF_MAX_DIMS]; // Steps in elements (not bytes) to next index
} mf_type_info;

static inline size_t mf_dtype_size(mf_dtype type) {
    switch(type) {
        case MF_DTYPE_F32: return 4;
        case MF_DTYPE_I32: return 4;
        case MF_DTYPE_U8:  return 1;
        default: return 0;
    }
}

/**
 * @brief Parses a string into an mf_dtype.
 * Case-insensitive, supports: "f32", "i32", "u8", "bool".
 */
mf_dtype mf_dtype_from_str(const char* s);

// --- Access Modes ---
typedef enum {
    MF_ACCESS_READ = 0,
    MF_ACCESS_WRITE = 1,
    MF_ACCESS_RW = 2
} mf_access_mode;

#endif // MF_TYPES_H

