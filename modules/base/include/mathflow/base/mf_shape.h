#ifndef MF_SHAPE_H
#define MF_SHAPE_H

#include <mathflow/base/mf_types.h>

/**
 * @brief Calculates total number of elements in a shape.
 */
size_t mf_shape_calc_count(const int32_t* shape, uint8_t ndim);

/**
 * @brief Calculates total bytes needed for a tensor.
 */
size_t mf_shape_calc_bytes(mf_dtype dtype, const int32_t* shape, uint8_t ndim);

/**
 * Checks if a shape is effectively a scalar (rank 0 or all dimensions are 1).
 */
bool mf_shape_is_scalar(const mf_type_info* info);

/**
 * Normalizes a shape by removing leading/trailing dimensions of size 1.
 * If target_rank is > 0, tries to reach that rank.
 */
void mf_shape_normalize(mf_type_info* info);

/**
 * @brief Calculates linear strides for a contiguous tensor based on its shape.
 */
void mf_shape_calc_strides(mf_type_info* info);

/**
 * @brief Formats a shape as a string (e.g. "[100, 200]").
 */
void mf_shape_format(const mf_type_info* info, char* buf, size_t size);

/**
 * Checks if two shapes can be broadcasted and returns the result in 'out'.
 * Returns true if successful, false if shapes are incompatible.
 */
bool   mf_shape_broadcast(const mf_type_info* a, const mf_type_info* b, mf_type_info* out);
i32    mf_shape_calc_linear_stride(size_t op_count, size_t dom_count);

#endif // MF_SHAPE_H
