#ifndef MF_STATE_H
#define MF_STATE_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/base/mf_memory.h>
#include <mathflow/base/mf_platform.h>

/**
 * @brief Persistent container for tensor data and memory management.
 * Owned by the Engine. Backends read from/write to this state.
 */
typedef struct mf_state {
    mf_tensor* registers;
    uint8_t* ownership_flags; // [register_count] 1 if owned, 0 if view
    size_t register_count;
    mf_allocator* allocator;
    
    // Backend-specific prepared execution plan
    void* baked_data;

    // Error flag set by execution contexts.
    // 0 = No Error. Uses mf_exec_error codes.
    mf_atomic_i32  error_code;
    mf_atomic_i32* global_error_ptr; // Points to engine->error_code for global Kill Switch
} mf_state;

#endif // MF_STATE_H