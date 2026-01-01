#ifndef MF_ENGINE_INTERNAL_H
#define MF_ENGINE_INTERNAL_H

#include <mathflow/engine/mf_engine.h>
#include <mathflow/isa/mf_state.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_backend.h>
#include <mathflow/engine/mf_pipeline.h>
#include <mathflow/base/mf_buffer.h>

/**
 * @brief Mapping between a Local Register in a Kernel and a Global Resource.
 */
typedef struct {
    u16 local_reg;   // Register index in the compiled program
    u16 global_res;  // Resource index in the engine's registry
    u8  flags;       // Symbol flags (Input, Output, etc.)
} mf_kernel_binding;

/**
 * @brief Relationship for automatic resizing (e.g. Out follows In).
 */
typedef struct {
    u16 src_res_idx;
    u16 dst_res_idx;
} mf_auto_resize_task;

/**
 * @brief Runtime instance of a Kernel (Program + State).
 */
typedef struct {
    const char* id;
    u32         id_hash;
    mf_program* program;
    mf_state    state;       // Local registers and memory
    uint32_t    frequency;   // Execution frequency per frame
    
    mf_kernel_binding* bindings;
    u32                binding_count;

    mf_auto_resize_task* resize_tasks;
    u32                  resize_task_count;
} mf_kernel_inst;

/**
 * @brief Concrete instance of a Global Resource (Double Buffered).
 */
typedef struct {
    const char* name;
    u32         name_hash;
    mf_buffer*  buffers[2];   // [0] Front, [1] Back
    size_t      size_bytes;
    mf_tensor   desc;         // Metadata and current view
} mf_resource_inst;

/**
 * @brief The Core Engine Structure.
 */
struct mf_engine {
    // Memory Management
    mf_arena arena;           // Static memory (Code, Metadata)
    void*    arena_buffer;
    mf_heap  heap;            // Dynamic memory (Tensors, Data)
    void*    heap_buffer;

    // Backend Implementation
    mf_backend backend;

    // Pipeline State
    mf_resource_inst* resources;
    u32               resource_count;
    mf_kernel_inst*   kernels;
    u32               kernel_count;

    // Buffer Synchronization
    u8 front_idx;             // Index for Read
    u8 back_idx;              // Index for Write
    
    // Stats
    uint64_t frame_index;
};

// --- Internal Utilities (Shared across module files) ---

/**
 * @brief Resets/Initializes the internal state for a kernel program.
 * Defined in mf_engine.c, used in mf_pipeline.c.
 */
void mf_state_reset(mf_state* state, const mf_program* prog, mf_arena* arena);

/**
 * @brief Finds resource index by its name hash.
 */
int32_t find_resource_idx(mf_engine* engine, u32 name_hash);

/**
 * @brief Finds symbol index in a program by its name hash.
 */
int32_t find_symbol_idx(const mf_program* prog, u32 name_hash);

#endif // MF_ENGINE_INTERNAL_H