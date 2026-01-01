#ifndef MF_ENGINE_INTERNAL_H
#define MF_ENGINE_INTERNAL_H

#include <mathflow/engine/mf_engine.h>
#include <mathflow/isa/mf_state.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_backend.h>

#include <mathflow/engine/mf_pipeline.h>
#include <mathflow/base/mf_buffer.h>

typedef struct {
    u16 local_reg;
    u16 global_res;
    u8 flags; // Cached MF_SYMBOL_FLAG_*
} mf_kernel_binding;

typedef struct {
    u16 src_res_idx;
    u16 dst_res_idx;
} mf_auto_resize_task;

// Instance of a loaded kernel within the engine
typedef struct {
    const char* id;
    u32 id_hash;
    mf_program* program;
    mf_state state; // Local registers for this kernel
    uint32_t frequency;
    
    // Cached mapping: Local Reg Index -> Global Resource Index
    mf_kernel_binding* bindings;
    u32 binding_count;

    // Pre-calculated auto-resize tasks
    mf_auto_resize_task* resize_tasks;
    u32 resize_task_count;
} mf_kernel_inst;

// Concrete implementation of a Global Resource instance
typedef struct {
    const char* name;
    u32 name_hash; // FNV-1a
    mf_buffer* buffers[2]; // Double buffering: [0] and [1]
    size_t size_bytes;
    mf_tensor desc; // Prototype (holds info and current buffer view)
} mf_resource_inst;

// The concrete implementation of the Engine.
struct mf_engine {
    mf_arena arena;
    void* arena_buffer;
    mf_backend backend;

    mf_heap heap;
    void* heap_buffer;
    
    mf_resource_inst* resources;
    u32 resource_count;
    
    mf_kernel_inst* kernels;
    u32 kernel_count;

    u8 front_idx;
    u8 back_idx;

    uint64_t frame_index;
};

#endif // MF_ENGINE_INTERNAL_H