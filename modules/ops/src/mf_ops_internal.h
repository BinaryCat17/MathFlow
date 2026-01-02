#ifndef MF_OPS_INTERNAL_H
#define MF_OPS_INTERNAL_H

#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_platform.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_exec_ctx.h>

void mf_ops_register_math(mf_op_func* table);
void mf_ops_register_logic(mf_op_func* table);
void mf_ops_register_matrix(mf_op_func* table);
void mf_ops_register_state(mf_op_func* table);

// Aggregate registrars (internal use only)
void mf_ops_core_register(mf_op_func* table);
void mf_ops_array_register(mf_op_func* table);

// --- Validation Macros ---

static inline bool _mf_should_log_error(mf_exec_ctx* ctx) {
    if (ctx->error != MF_ERROR_NONE) return false;
    if (ctx->global_error_ptr && mf_atomic_load(ctx->global_error_ptr) != 0) return false;
    return true;
}

// Validates Input Tensor: Must be completely valid (Struct + Buffer + Data + non-zero size)
#define MF_CHECK_INPUT(CTX, T) \
    do { \
        if (!(T) || !(T)->buffer || !(T)->buffer->data || (T)->buffer->size_bytes == 0) { \
            if (_mf_should_log_error(CTX)) { \
                int reg_idx = -1; \
                if ((T) >= (CTX)->registers && (T) < (CTX)->registers + (CTX)->register_count) { \
                    reg_idx = (int)((T) - (CTX)->registers); \
                } \
                MF_LOG_ERROR("Runtime Error: Invalid INPUT tensor access (Reg: %d, Unallocated, Null or Zero Size). Op execution aborted.", reg_idx); \
            } \
            (CTX)->error = MF_ERROR_RUNTIME; \
            return; \
        } \
    } while(0)

// Validates Destination View: The tensor handle itself must exist, but data can be null (pre-allocation)
#define MF_CHECK_DST_VIEW(CTX, T) \
    do { \
        if (!(T)) { \
            if (_mf_should_log_error(CTX)) { \
                MF_LOG_ERROR("Runtime Error: Invalid DST tensor handle (NULL). Op execution aborted."); \
            } \
            (CTX)->error = MF_ERROR_RUNTIME; \
            return; \
        } \
    } while(0)

// Validates Destination Data: Must be called AFTER allocation/resize
#define MF_CHECK_DST_DATA(CTX, T) \
    do { \
        if (!(T) || !(T)->buffer || !(T)->buffer->data || (T)->buffer->size_bytes == 0) { \
            if (_mf_should_log_error(CTX)) { \
                int reg_idx = -1; \
                if ((T) >= (CTX)->registers && (T) < (CTX)->registers + (CTX)->register_count) { \
                    reg_idx = (int)((T) - (CTX)->registers); \
                } \
                MF_LOG_ERROR("Runtime Error: Invalid DST tensor data (Reg: %d, Allocation failed or size is 0). Op execution aborted.", reg_idx); \
            } \
            (CTX)->error = MF_ERROR_OOM; \
            return; \
        } \
    } while(0)

// Generic Pointer Check
#define MF_CHECK_PTR(CTX, PTR) \
    do { \
        if (!(PTR)) { \
            if (_mf_should_log_error(CTX)) { \
                MF_LOG_ERROR("Runtime Error: Internal pointer is NULL. Op execution aborted."); \
            } \
            (CTX)->error = MF_ERROR_RUNTIME; \
            return; \
        } \
    } while(0)

#endif // MF_OPS_INTERNAL_H
