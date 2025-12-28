#ifndef MF_DISPATCH_TABLE_H
#define MF_DISPATCH_TABLE_H

#include <mathflow/isa/mf_kernel_ctx.h>
#include <mathflow/isa/mf_opcodes.h>

// --- Backend Interface ---

// Hook for sync (e.g. GPU upload/download)
// Called by the runtime (VM) when a tensor is mapped.
typedef void (*mf_hook_map)(void* impl, mf_tensor* tensor, mf_access_mode mode);

// The Dispatch Table
// Connects the VM Opcodes to the Kernel Implementations.
typedef struct {
    mf_op_func op_table[MF_OP_LIMIT];
    mf_hook_map on_map;
} mf_backend_dispatch_table;

#endif // MF_DISPATCH_TABLE_H
