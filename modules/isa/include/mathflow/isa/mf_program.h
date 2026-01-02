#ifndef MF_PROGRAM_H
#define MF_PROGRAM_H

#include <mathflow/base/mf_types.h>
#include "mf_instruction.h"
#include "mf_tensor.h"

#define MF_BINARY_MAGIC 0x4D464C57 // "MFLW"
#define MF_BINARY_VERSION 12       // Removed Identity (Shape-Driven ISA)

#define MF_MAX_SYMBOL_NAME 64

// Symbol Flags
#define MF_SYMBOL_FLAG_INPUT  (1 << 0) // Read-Only (Bind to Front Buffer)
#define MF_SYMBOL_FLAG_OUTPUT (1 << 1) // Write-Only (Bind to Back Buffer)

// Map Name -> Register Index
typedef struct {
    char name[MF_MAX_SYMBOL_NAME];
    char provider[MF_MAX_SYMBOL_NAME];
    uint32_t name_hash; // FNV-1a
    uint32_t register_idx;
    uint32_t related_name_hash; // Hash of the Input symbol that drives this Output's shape (0 if none)
    uint8_t flags;       // MF_SYMBOL_FLAG_*
    uint8_t reserved[3]; // Padding/Alignment
} mf_bin_symbol;

// A single execution unit within a program (e.g. for a specific Output shape)
typedef struct {
    uint32_t start_inst;
    uint32_t inst_count;
    uint32_t domain_reg; // Index of the register that defines the execution domain (usually an Output)
    uint32_t reserved;
} mf_task;

// Metadata for a single tensor in the binary file
// Followed immediately by shape data? No, fixed max dims.
typedef struct {
    uint8_t dtype;       // mf_dtype
    uint8_t ndim;        // Rank
    uint8_t is_constant; // 1 if data follows, 0 if uninitialized buffer
    uint8_t reserved;    // Padding
    
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
    u32 version;           // 7
    
    u32 instruction_count; 
    u32 tensor_count;      // Total number of registers/tensors
    u32 symbol_count;      // Number of named I/O entries
    u32 task_count;        // Number of execution tasks
    
    u32 reserved[6];       
} mf_bin_header;

// In-memory representation
typedef struct mf_program {
    mf_bin_header meta;
    
    mf_instruction* code;
    
    // Array of tensor descriptors initialized from file.
    // NOTE: 'data' pointers here point to the Program's Constant Data Block.
    // When VM loads this, it clones these tensors into its own memory pool.
    mf_tensor* tensors; 
    
    mf_bin_symbol* symbols;
    mf_task* tasks;
} mf_program;

#endif // MF_PROGRAM_H
