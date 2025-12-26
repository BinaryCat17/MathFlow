#ifndef MF_BASE_H
#define MF_BASE_H

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

#endif // MF_BASE_H
