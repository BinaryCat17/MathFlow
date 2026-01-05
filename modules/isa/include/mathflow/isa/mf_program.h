#ifndef MF_PROGRAM_H
#define MF_PROGRAM_H

#include <mathflow/base/mf_types.h>
#include "mf_instruction.h"
#include "mf_tensor.h"

#define MF_BINARY_MAGIC   0x4D464C57 // "MFLW"
#define MF_BINARY_VERSION 20         // Phase 9: The Cartridge Model (Container Format)

#define MF_MAX_SYMBOL_NAME 64
#define MF_MAX_TITLE_NAME 128
#define MF_MAX_SECTIONS    16

// Section Types
typedef enum {
    MF_SECTION_PROGRAM  = 0x01, // Compiled MathFlow Bytecode
    MF_SECTION_PIPELINE = 0x02, // Execution schedule and resource bindings (JSON)
    MF_SECTION_IMAGE    = 0x03, // Embedded Texture (Raw or Compressed)
    MF_SECTION_FONT     = 0x04, // Embedded SDF Font Data
    MF_SECTION_RAW      = 0x05, // Arbitrary data blob
} mf_section_type;

// Symbol Flags (for Port Mapping)
#define MF_SYMBOL_FLAG_INPUT  (1 << 6) // Read-Only (Bind to Front Buffer)
#define MF_SYMBOL_FLAG_OUTPUT (1 << 7) // Write-Only (Bind to Back Buffer)

// Tensor Flags
#define MF_TENSOR_FLAG_CONSTANT   (1 << 0)
#define MF_TENSOR_FLAG_REDUCTION  (1 << 1)
#define MF_TENSOR_FLAG_GENERATOR  (1 << 2)
#define MF_TENSOR_FLAG_ALIAS      (1 << 3) // Bound to external resource (Input/Output)

// Binding Flags
#define MF_BINDING_FLAG_REDUCTION (1 << 0)

// --- Cartridge Container (Level 0) ---

typedef struct {
    char name[MF_MAX_SYMBOL_NAME];
    uint32_t type;   // mf_section_type
    uint32_t offset; // Offset from start of file
    uint32_t size;   // Size in bytes
    uint32_t reserved[4];
} mf_section_header;

typedef struct {
    u32 magic;             // 0x4D464C57
    u32 version;           // MF_BINARY_VERSION
    
    // App Settings
    char app_title[MF_MAX_TITLE_NAME];
    u32 window_width;
    u32 window_height;
    u32 num_threads;       // 0 = Auto
    u8 vsync;              // 1 = Enabled
    u8 fullscreen;         // 1 = Enabled
    u8 resizable;          // 1 = Enabled
    u8 reserved_flags[1];

    u32 section_count;
    mf_section_header sections[MF_MAX_SECTIONS];

    u32 reserved[8];       
} mf_cartridge_header;

// --- Program Section (Level 1) ---

// Map Name -> Register Index
typedef struct {
    char name[MF_MAX_SYMBOL_NAME];
    char provider[MF_MAX_SYMBOL_NAME];
    uint32_t name_hash; // FNV-1a
    uint32_t register_idx;
    uint32_t related_name_hash; // Hash of the Input symbol that drives this Output's shape (0 if none)
    uint8_t flags;       // MF_SYMBOL_FLAG_* | MF_RESOURCE_FLAG_*
    uint16_t builtin_id; // mf_builtin_id
    uint8_t builtin_axis; // For indexed providers like host.index.N
    uint8_t reserved[1];
} mf_bin_symbol;

// Binding between a register and a task's domain
typedef struct {
    uint16_t reg_idx;
    uint16_t flags;      // MF_BINDING_FLAG_*
    int32_t byte_stride; // Pre-calculated: stride * sizeof(dtype)
} mf_bin_task_binding;

// A single execution unit within a program (e.g. for a specific Output shape)
typedef struct {
    uint32_t start_inst;
    uint32_t inst_count;
    uint32_t domain_reg; // Index of the register that defines the execution domain (usually an Output)
    uint8_t strategy;    // mf_dispatch_strategy
    uint8_t reserved[3];
    
    uint32_t binding_offset; // Offset into global binding table
    uint32_t binding_count;  // Number of registers used in this task
} mf_task;

// Metadata for a single tensor in the binary file
typedef struct {
    uint8_t dtype;       // mf_dtype
    uint8_t ndim;        // Rank
    uint8_t is_constant; // 1 if data follows, 0 if uninitialized buffer
    uint8_t builtin_id;  // mf_builtin_id (0 if none)
    uint8_t builtin_axis; // Axis for indexed providers (e.g. host.index.N)
    uint8_t flags;       // MF_TENSOR_FLAG_*
    uint8_t reserved[2]; // Padding
    
    int32_t shape[MF_MAX_DIMS];
    
    uint64_t data_size;  // Size in bytes of the initial data (0 if not constant)
} mf_bin_tensor_desc;

// Header for a PROGRAM section
typedef struct {
    u32 instruction_count; 
    u32 tensor_count;      // Total number of registers/tensors
    u32 symbol_count;      // Number of named I/O entries (Resource Templates)
    u32 task_count;        // Number of execution tasks
    u32 binding_count;     // Total number of register bindings
    
    u32 reduction_scratch_size; // Elements needed for reductions
    u32 sync_scratch_size;      // Elements needed for sync operations
    
    u32 reserved[8];       
} mf_bin_header;

// In-memory representation of a single program
typedef struct mf_program {
    mf_bin_header meta;
    
    mf_instruction* code;
    
    // Array of descriptors and initial constant data
    mf_type_info* tensor_infos;
    void** tensor_data;
    
    uint8_t* builtin_ids;  // Array of mf_builtin_id per tensor
    uint8_t* builtin_axes; // Array of builtin axis per tensor
    uint8_t* tensor_flags; // Array of tensor flags

    mf_bin_symbol* symbols;
    mf_task* tasks;
    mf_bin_task_binding* bindings;
} mf_program;

#endif // MF_PROGRAM_H
