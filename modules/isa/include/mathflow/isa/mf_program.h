#ifndef MF_PROGRAM_H
#define MF_PROGRAM_H

#include "mf_base.h"
#include "mf_instruction.h"
#include "mf_tensor.h"

#define MF_BINARY_MAGIC 0x4D464C57 // "MFLW"
#define MF_BINARY_VERSION 4        // Bump for Tensor Era

// Metadata for a single tensor in the binary file
// Followed immediately by shape data? No, fixed max dims.
typedef struct {
    uint8_t dtype;       // mf_dtype
    uint8_t ndim;        // Rank
    uint8_t is_constant; // 1 if data follows, 0 if uninitialized buffer
    uint8_t reserved;    
    
    int32_t shape[MF_MAX_DIMS];
    
    uint64_t data_size;  // Size in bytes of the initial data (0 if not constant)
    // In file: Raw data follows immediately after this struct if is_constant=1?
    // Better: All descriptors first, then all raw data blob. 
    // So we need offset? Or just implicit order.
    // Let's use implicit order to simplify reading.
} mf_bin_tensor_desc;

// File Header for .bin files
typedef struct {
    u32 magic;             // 0x4D464C57
    u32 version;           // 4
    
    u32 instruction_count; 
    u32 tensor_count;      // Total number of registers/tensors
    
    u32 reserved[8];       
} mf_bin_header;

// In-memory representation
typedef struct {
    mf_bin_header meta;
    
    mf_instruction* code;
    
    // Array of tensor descriptors initialized from file.
    // NOTE: 'data' pointers here point to the Program's Constant Data Block.
    // When VM loads this, it clones these tensors into its own memory pool.
    mf_tensor* tensors; 
} mf_program;

#endif // MF_PROGRAM_H
