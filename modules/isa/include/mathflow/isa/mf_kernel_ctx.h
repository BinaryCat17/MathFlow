#ifndef MF_KERNEL_CTX_H
#define MF_KERNEL_CTX_H

#include <mathflow/isa/mf_tensor.h>

// --- Access Modes ---

typedef enum {
    MF_ACCESS_READ = 0,
    MF_ACCESS_WRITE = 1,
    MF_ACCESS_RW = 2
} mf_access_mode;

// --- Kernel Context Interface ---

typedef struct mf_kernel_ctx mf_kernel_ctx;

struct mf_kernel_ctx {
    // Opaque pointer to the runtime implementation (e.g., mf_vm)
    void* impl; 

    /**
     * @brief Maps a tensor by its register index.
     * @param impl The runtime implementation.
     * @param idx The register index.
     * @param mode Access intent (Read/Write).
     * @return Pointer to the tensor, or NULL if invalid.
     */
    mf_tensor* (*map_tensor)(void* impl, u16 idx, mf_access_mode mode);

    /**
     * @brief Resizes a tensor.
     * @param impl The runtime implementation.
     * @param tensor The tensor to resize.
     * @param shape New dimensions array.
     * @param ndim Number of dimensions.
     * @return true if successful, false on error (OOM).
     */
    bool (*resize_tensor)(void* impl, mf_tensor* tensor, const int32_t* shape, uint8_t ndim);

    /**
     * @brief Reports a runtime error.
     * @param impl The runtime implementation.
     * @param error_code Implementation-specific error code.
     */
    void (*error)(void* impl, int error_code);
};

// --- Kernel Signature ---

// Standard signature for all math kernels.
// Replaces the old (mf_vm* vm, ...) signature.
typedef void (*mf_op_func)(const mf_kernel_ctx* ctx, u16 dest, u16 src1, u16 src2);

#endif // MF_KERNEL_CTX_H
