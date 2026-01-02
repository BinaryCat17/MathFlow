#ifndef MF_OPS_INTERNAL_H
#define MF_OPS_INTERNAL_H

#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_platform.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_exec_ctx.h>

/**
 * Internal helpers for MathFlow kernels.
 */

static inline bool _mf_should_log_error(mf_exec_ctx* ctx) {
    if (ctx->error != MF_ERROR_NONE) return false;
    if (ctx->global_error_ptr && mf_atomic_load(ctx->global_error_ptr) != 0) return false;
    return true;
}

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