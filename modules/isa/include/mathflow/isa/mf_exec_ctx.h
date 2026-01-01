#ifndef MF_EXEC_CTX_H
#define MF_EXEC_CTX_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/base/mf_memory.h>
#include <string.h> // memset

// Forward decl
typedef struct mf_exec_ctx mf_exec_ctx;

// --- Execution State ---
typedef enum {
    MF_ERROR_NONE = 0,
    MF_ERROR_OOM = 1,          
    MF_ERROR_SHAPE_MISMATCH = 2, 
    MF_ERROR_INVALID_OP = 3,
    MF_ERROR_RUNTIME = 4
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
    
    // N-Dimensional Context
    u8 ndim;
    u32 tile_offset[MF_MAX_DIMS];  // Start coords of this tile/batch
    u32 tile_size[MF_MAX_DIMS];    // Size of this tile/batch (active elements)
    u32 domain_shape[MF_MAX_DIMS]; // Total size of the execution domain

    // State
    mf_exec_error error;
    
    // User Data
    void* user_data;
};

// --- Execution Context API (Inlined) ---

static inline void mf_exec_ctx_init(mf_exec_ctx* ctx, mf_tensor* registers, size_t reg_count, mf_allocator* allocator) {
    memset(ctx, 0, sizeof(mf_exec_ctx));
    ctx->registers = registers;
    ctx->register_count = reg_count;
    ctx->allocator = allocator;
    ctx->ndim = 1; // Default
    ctx->tile_size[0] = 1;
    ctx->domain_shape[0] = 1;
    ctx->batch_size = 1;
}

static inline mf_tensor* mf_exec_ctx_map_tensor(mf_exec_ctx* ctx, u16 idx, mf_access_mode mode) {
    (void)mode;
    if (idx >= ctx->register_count) return NULL;
    return &ctx->registers[idx];
}

static inline bool mf_exec_ctx_resize_tensor(mf_exec_ctx* ctx, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim) {
    if (!tensor) return false;
    
    mf_type_info info;
    info.dtype = tensor->info.dtype; // Preserve type
    info.ndim = new_ndim;
    if (new_ndim > 0) memcpy(info.shape, new_shape, sizeof(int32_t) * new_ndim);
    
    // Default Strides
    int32_t stride = 1;
    for (int k = new_ndim - 1; k >= 0; --k) {
        info.strides[k] = stride;
        stride *= info.shape[k];
    }

    if (!mf_tensor_resize(tensor, ctx->allocator, &info)) {
        ctx->error = MF_ERROR_OOM;
        return false;
    }
    return true;
}

#endif // MF_EXEC_CTX_H