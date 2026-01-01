#ifndef MF_EXEC_CTX_H
#define MF_EXEC_CTX_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/base/mf_memory.h>
#include <mathflow/base/mf_platform.h>
#include <string.h> // memset

// Forward decl
typedef struct mf_exec_ctx mf_exec_ctx;

// --- Execution State ---
typedef enum {
    MF_ERROR_NONE = 0,
    MF_ERROR_OOM = 1,          
    MF_ERROR_SHAPE_MISMATCH = 2, 
    MF_ERROR_INVALID_OP = 3,
    MF_ERROR_RUNTIME = 4,
    MF_ERROR_OUT_OF_BOUNDS = 5
} mf_exec_error;

static inline const char* mf_exec_error_to_str(mf_exec_error err) {
    switch (err) {
        case MF_ERROR_NONE:           return "NONE";
        case MF_ERROR_OOM:            return "OUT_OF_MEMORY";
        case MF_ERROR_SHAPE_MISMATCH: return "SHAPE_MISMATCH";
        case MF_ERROR_INVALID_OP:     return "INVALID_OPCODE";
        case MF_ERROR_RUNTIME:        return "RUNTIME_GENERIC_ERROR";
        case MF_ERROR_OUT_OF_BOUNDS:  return "OUT_OF_BOUNDS";
        default:                      return "UNKNOWN_ERROR";
    }
}

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
    u32 linear_offset;             // Linear start index of this tile
    u32 tile_offset[MF_MAX_DIMS];  // Start coords of this tile/batch
    u32 tile_size[MF_MAX_DIMS];    // Size of this tile/batch (active elements)
    u32 domain_shape[MF_MAX_DIMS]; // Total size of the execution domain

    // State
    mf_exec_error error;
    mf_atomic_i32* global_error_ptr;
    
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
    ctx->global_error_ptr = NULL;
}

static inline mf_tensor* mf_exec_ctx_map_tensor(mf_exec_ctx* ctx, u16 idx, mf_access_mode mode) {
    (void)mode;
    if (idx >= ctx->register_count) return NULL;
    return &ctx->registers[idx];
}

static inline bool mf_exec_ctx_resize_tensor(mf_exec_ctx* ctx, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim) {
    if (!tensor) return false;
    
    mf_type_info info;
    mf_type_info_init_contiguous(&info, tensor->info.dtype, new_shape, new_ndim);

    if (!mf_tensor_resize(tensor, ctx->allocator, &info)) {
        ctx->error = MF_ERROR_OOM;
        return false;
    }
    return true;
}

/**
 * @brief Allocates temporary memory from the thread-local scratchpad.
 * This memory is only valid during the current instruction execution or tile processing.
 */
static inline void* mf_exec_ctx_scratch_alloc(mf_exec_ctx* ctx, size_t size) {
    if (!ctx->allocator) return NULL;
    return ctx->allocator->alloc(ctx->allocator, size);
}

/**
 * @brief Creates a temporary tensor on the scratchpad.
 */
static inline mf_tensor* mf_exec_ctx_scratch_tensor(mf_exec_ctx* ctx, const mf_type_info* info) {
    if (!ctx->allocator) return NULL;
    mf_tensor* t = (mf_tensor*)ctx->allocator->alloc(ctx->allocator, sizeof(mf_tensor));
    if (!t) return NULL;
    if (!mf_tensor_alloc(t, ctx->allocator, info)) return NULL;
    return t;
}

#endif // MF_EXEC_CTX_H