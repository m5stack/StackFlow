#include <vector>
#include <cstddef> // For size_t
#include <stdexcept> // For std::invalid_argument
#include <algorithm> // For std::copy
#include <type_traits> // For std::enable_if, std::is_arithmetic

/**
 * @brief Concatenates two 3D arrays along the last dimension (dim=2).
 *        Input and output data are represented as 1D std::vectors in row-major order.
 *
 * @tparam T The data type of the elements (e.g., float, int).
 *          Must be an arithmetic type.
 * @param hift_cache_mel_data The first input 1D vector (data for hift_cache_mel).
 * @param dim0_h The size of the first dimension for hift_cache_mel.
 * @param dim1_h The size of the second dimension for hift_cache_mel.
 * @param dim2_h The size of the third dimension for hift_cache_mel.
 * @param tts_mel_data The second input 1D vector (data for tts_mel).
 * @param dim0_t The size of the first dimension for tts_mel.
 * @param dim1_t The size of the second dimension for tts_mel.
 * @param dim2_t The size of the third dimension for tts_mel.
 * @return std::vector<T> A new 1D vector containing the concatenated data.
 * @throws std::invalid_argument If dimensions are incompatible.
 */
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
concat_3d_dim2(const std::vector<T>& hift_cache_mel_data,
               size_t dim0_h, size_t dim1_h, size_t dim2_h,
               const std::vector<T>& tts_mel_data,
               size_t dim0_t, size_t dim1_t, size_t dim2_t) {

    // --- Input Validation ---
    if (hift_cache_mel_data.size() != dim0_h * dim1_h * dim2_h) {
        throw std::invalid_argument("hift_cache_mel_data size does not match provided dimensions.");
    }
    if (tts_mel_data.size() != dim0_t * dim1_t * dim2_t) {
        throw std::invalid_argument("tts_mel_data size does not match provided dimensions.");
    }
    if (dim0_h != dim0_t || dim1_h != dim1_t) {
        throw std::invalid_argument("First two dimensions (dim0, dim1) of input arrays must match for concatenation along dim=2.");
    }

    const size_t dim0 = dim0_h; // == dim0_t
    const size_t dim1 = dim1_h; // == dim1_t
    const size_t dim2_result = dim2_h + dim2_t;

    // Handle case where result would be empty
    if (dim0 == 0 || dim1 == 0 || dim2_result == 0) {
        return std::vector<T>();
    }

    // --- Calculate Result Size and Reserve Memory ---
    const size_t result_size = dim0 * dim1 * dim2_result;
    std::vector<T> result_data;
    result_data.reserve(result_size); // Pre-allocate for efficiency

    // --- Calculate Strides (Row-Major Order) ---
    const size_t stride_dim2_h = 1;
    const size_t stride_dim1_h = dim2_h * stride_dim2_h;
    const size_t stride_dim0_h = dim1_h * stride_dim1_h;

    const size_t stride_dim2_t = 1;
    const size_t stride_dim1_t = dim2_t * stride_dim2_t;
    const size_t stride_dim0_t = dim1_t * stride_dim1_t;

    const size_t stride_dim2_result = 1;
    const size_t stride_dim1_result = dim2_result * stride_dim2_result;
    const size_t stride_dim0_result = dim1 * stride_dim1_result; // dim1 == dim1_h == dim1_t

    // --- Perform Concatenation ---
    result_data.resize(result_size); // Resize once for direct access
    size_t result_idx = 0;

    for (size_t n = 0; n < dim0; ++n) {
        for (size_t c = 0; c < dim1; ++c) {
            // --- Copy slice from hift_cache_mel [n][c][:] ---
            const size_t hift_base_offset = n * stride_dim0_h + c * stride_dim1_h;
            const size_t hift_slice_end_offset = hift_base_offset + dim2_h * stride_dim2_h;
            // Use std::copy for efficient block copy
            std::copy(hift_cache_mel_data.data() + hift_base_offset,
                      hift_cache_mel_data.data() + hift_slice_end_offset,
                      result_data.data() + result_idx);
            result_idx += dim2_h;

            // --- Copy slice from tts_mel [n][c][:] ---
            const size_t tts_base_offset = n * stride_dim0_t + c * stride_dim1_t;
            const size_t tts_slice_end_offset = tts_base_offset + dim2_t * stride_dim2_t;
            // Use std::copy for efficient block copy
            std::copy(tts_mel_data.data() + tts_base_offset,
                      tts_mel_data.data() + tts_slice_end_offset,
                      result_data.data() + result_idx);
            result_idx += dim2_t;
        }
    }

    return result_data; // RVO should apply
}