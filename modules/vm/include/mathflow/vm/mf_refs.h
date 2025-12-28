#ifndef MF_REFS_H
#define MF_REFS_H

#include <mathflow/base/mf_types.h>

// Represents a reference to a tensor value (Scalar or Slice)
// These structures abstract pointers to data.
// In the future, they could hold handles, offsets, or debug info.

typedef struct { f32* p; } mf_ref_f32;
typedef struct { mf_vec2* p; } mf_ref_vec2;
typedef struct { mf_vec3* p; } mf_ref_vec3;
typedef struct { mf_vec4* p; } mf_ref_vec4;
typedef struct { mf_mat3* p; } mf_ref_mat3;
typedef struct { mf_mat4* p; } mf_ref_mat4;
typedef struct { u8* p; } mf_ref_bool;

// --- Null Constants ---
static const mf_ref_f32 MF_NULL_F32 = { 0 };
static const mf_ref_vec2 MF_NULL_VEC2 = { 0 };
static const mf_ref_vec3 MF_NULL_VEC3 = { 0 };
static const mf_ref_vec4 MF_NULL_VEC4 = { 0 };
static const mf_ref_mat3 MF_NULL_MAT3 = { 0 };
static const mf_ref_mat4 MF_NULL_MAT4 = { 0 };
static const mf_ref_bool MF_NULL_BOOL = { 0 };

// --- Accessors ---

// Check if reference is valid
#define MF_VALID(ref) ((ref).p != NULL)

// Dereference (Read/Write)
// Usage: MF_VAL(my_ref) = 5.0f;
#define MF_VAL(ref) (*(ref).p)

#endif // MF_REFS_H
