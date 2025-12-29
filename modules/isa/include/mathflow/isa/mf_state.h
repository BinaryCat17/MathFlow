#ifndef MF_STATE_H
#define MF_STATE_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/base/mf_memory.h>

/**
 * @brief Persistent container for tensor data and memory management.
 * Owned by the Engine. Backends read from/write to this state.
 */
typedef struct mf_state {
    mf_tensor* registers;
    size_t register_count;
    mf_allocator* allocator;
} mf_state;

#endif // MF_STATE_H