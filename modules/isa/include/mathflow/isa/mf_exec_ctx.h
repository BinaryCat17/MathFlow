#ifndef MF_EXEC_CTX_H
#define MF_EXEC_CTX_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/base/mf_memory.h>

// Forward decl
typedef struct mf_exec_ctx mf_exec_ctx;

// --- Execution State ---
typedef enum {
    MF_ERROR_NONE = 0,
    MF_ERROR_OOM = 1,          
    MF_ERROR_SHAPE_MISMATCH = 2, 
    MF_ERROR_INVALID_OP = 3    
} mf_exec_error;

/**
 * @brief Light-weight execution context (Ephemeral).
 * Created on the stack or per-thread. Points to data in mf_state or tiled buffers.
 */
struct mf_exec_ctx {
    // View of Registers (Can point to persistent state or local thread-local tensors)
    mf_tensor* registers;
    size_t register_count;

    // Optional allocator for temporary allocations during execution
    mf_allocator* allocator; 
    
    // Execution Configuration
    u32 batch_size; 
    u32 global_offset[3];
    u32 local_size[3];
    u32 global_size[3];

    // State
    mf_exec_error error;
    
    // User Data
    void* user_data;
};

// --- Execution Context API ---

// Initialize a context instance (lightweight, no allocations)
void mf_exec_ctx_init(mf_exec_ctx* ctx, mf_tensor* registers, size_t reg_count, mf_allocator* allocator);

// --- Accessors ---
// Returns a pointer to the live tensor in the context.
mf_tensor* mf_exec_ctx_map_tensor(mf_exec_ctx* ctx, u16 idx, mf_access_mode mode);

// Resizes a tensor (if allocator is present)
bool mf_exec_ctx_resize_tensor(mf_exec_ctx* ctx, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim);

#endif // MF_EXEC_CTX_H
