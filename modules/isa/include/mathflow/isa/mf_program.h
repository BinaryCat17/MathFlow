#ifndef MF_PROGRAM_H
#define MF_PROGRAM_H

#include "mf_base.h"
#include "mf_instruction.h"

#define MF_BINARY_MAGIC 0x4D464C57 // "MFLW"
#define MF_BINARY_VERSION 3        // Version bumped for mat3

// File Header for .bin files
typedef struct {
    u32 magic;             // 0x4D464C57
    u32 version;           // 3
    
    u32 instruction_count; // Number of instructions in the code section
    
    // Data Section Counts
    u32 f32_count;
    u32 vec2_count;
    u32 vec3_count;
    u32 vec4_count;
    u32 mat3_count;
    u32 mat4_count;
    u32 bool_count; // stored as u8
    
    u32 reserved[8];       // Padding/Reserved for future flags
} mf_bin_header;

// In-memory representation of a program ready to be loaded by the VM
typedef struct {
    mf_bin_header meta;
    
    mf_instruction* code; // Array of instructions
    
    // Data Pointers (Points to the start of each data block)
    f32*     data_f32;
    mf_vec2* data_vec2;
    mf_vec3* data_vec3;
    mf_vec4* data_vec4;
    mf_mat3* data_mat3;
    mf_mat4* data_mat4;
    u8*      data_bool;
} mf_program;

#endif // MF_PROGRAM_H