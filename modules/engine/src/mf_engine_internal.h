#ifndef MF_ENGINE_INTERNAL_H
#define MF_ENGINE_INTERNAL_H

#include <mathflow/engine/mf_engine.h>
#include <mathflow/isa/mf_state.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_backend.h>

// Internal State Buffer Pair
typedef struct {
    void* buffer_a;
    void* buffer_b;
    size_t size;
} mf_state_buffer;

// The concrete implementation of the Engine.
// Combines Static Resources (Code) and Execution State (Data).
struct mf_engine {
    // Static Resources
    mf_arena arena;
    void* arena_buffer;
    mf_program* program;
    mf_backend backend;

    // Execution State (Single Source of Truth)
    mf_state state;
    mf_heap heap;
    void* heap_buffer;
    
    // Global Config (used for dispatch)
    u32 global_size[3];
    
    // State Management (Double Buffering)
    mf_state_buffer* state_buffers;
    uint64_t frame_index;
};

#endif // MF_ENGINE_INTERNAL_H