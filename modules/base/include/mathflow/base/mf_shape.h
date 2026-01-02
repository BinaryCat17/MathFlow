#ifndef MF_SHAPE_H
#define MF_SHAPE_H

#include <mathflow/base/mf_types.h>

/**
 * Checks if two shapes can be broadcasted and returns the result in 'out'.
 * Returns true if successful, false if shapes are incompatible.
 */
bool mf_shape_broadcast(const mf_type_info* a, const mf_type_info* b, mf_type_info* out);

/**
 * Calculates contiguous strides for a given shape.
 */
void mf_shape_calc_strides(mf_type_info* info);

/**
 * @brief Infers strides for a tensor when executed within a specific domain.
 * Implements Smart Broadcasting (NumPy style).
 * 
 * @param shape The shape of the tensor.
 * @param domain The shape of the execution domain.
 * @param out_strides Array to fill with inferred strides (size MF_MAX_DIMS).
 */
void mf_shape_infer_strides(const mf_type_info* shape, const mf_type_info* domain, int32_t* out_strides);

/**
 * Formats shape as a string "[dim0, dim1, ...]"
 */
void mf_shape_format(const mf_type_info* info, char* buf, size_t size);

/**
 * Calculates the linear element-stride of an operand relative to a domain.
 * Used for the STEP_N execution model.
 */
i32 mf_shape_calc_linear_stride(size_t op_count, size_t dom_count);

/**
 * Validates if a resource shape can be bound to a kernel port.
 */
bool mf_shape_is_compatible(const mf_type_info* port, const mf_type_info* res, bool is_output);

#endif // MF_SHAPE_H
