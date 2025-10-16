// src/slice_3d.tpp
#ifndef SLICE_3D_TPP
#define SLICE_3D_TPP

#include "slice_3d.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip> // For std::setprecision

// --- Helper for index normalization ---
// Handles ALL negative indices, including -1, according to Python rules.
// -1 means the last element, -2 the second to last, etc.
template <typename T>
typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, size_t>::type
normalize_slice_index(T index, size_t dim_size) {
    if (index < 0) {
        // In Python, negative indices are relative to the end.
        // e.g., -1 is the last element, -2 is second to last.
        // We clamp the result to be >= 0.
        return static_cast<size_t>(std::max(static_cast<T>(0), static_cast<T>(dim_size) + index));
    } else {
        // Clamp positive indices to dim_size (Python allows index==dim_size for stop)
        return static_cast<size_t>(std::min(static_cast<size_t>(index), dim_size));
    }
}

// --- Main optimized slicing function ---
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_optimized(std::vector<T>& data_1d,
                   size_t dim0, size_t dim1, size_t dim2,
                   int start0, int stop0,
                   int start1, int stop1,
                   int start2, int stop2) {

    // --- Input Validation ---
    const size_t total_elements = dim0 * dim1 * dim2;
    if (data_1d.size() != total_elements) {
        throw std::invalid_argument("Data size does not match provided dimensions.");
    }

    // --- Normalize ALL Indices using the same logic ---
    // This correctly handles ALL negative indices, including stop==-1.
    // There is NO special case for stop==-1 anymore.
    const size_t norm_start0 = normalize_slice_index(start0, dim0);
    const size_t norm_stop0 = normalize_slice_index(stop0, dim0); // Treat -1 as "last element index"

    const size_t norm_start1 = normalize_slice_index(start1, dim1);
    const size_t norm_stop1 = normalize_slice_index(stop1, dim1); // Treat -1 as "last element index"

    const size_t norm_start2 = normalize_slice_index(start2, dim2);
    const size_t norm_stop2 = normalize_slice_index(stop2, dim2); // Treat -1 as "last element index"

    // --- Calculate Final Slice Boundaries ---
    // Ensure start <= stop (handles user errors like [5:3] and empty slices like [3:3])
    // Python slicing returns an empty list if start >= stop.
    // const size_t final_start0 = std::min(norm_start0, norm_stop0);
    // const size_t final_stop0 = std::max(norm_start0, norm_stop0); // Ensure stop >= start

    // const size_t final_start1 = std::min(norm_start1, norm_stop1);
    // const size_t final_stop1 = std::max(norm_start1, norm_stop1); // Ensure stop >= start

    // const size_t final_start2 = std::min(norm_start2, norm_stop2);
    // const size_t final_stop2 = std::max(norm_start2, norm_stop2); // Ensure stop >= start

    // --- Calculate Effective Slice Lengths ---
    const size_t final_start0 = norm_start0;
    const size_t final_stop0 = norm_stop0;
    const size_t slice_len0 = (final_start0 < final_stop0) ? (final_stop0 - final_start0) : 0;

    const size_t final_start1 = norm_start1;
    const size_t final_stop1 = norm_stop1;
    const size_t slice_len1 = (final_start1 < final_stop1) ? (final_stop1 - final_start1) : 0;

    const size_t final_start2 = norm_start2;
    const size_t final_stop2 = norm_stop2;
    const size_t slice_len2 = (final_start2 < final_stop2) ? (final_stop2 - final_start2) : 0;

    // --- Handle Empty Result ---
    if (slice_len0 == 0 || slice_len1 == 0 || slice_len2 == 0) {
        return std::vector<T>();
    }

    // --- Calculate Result Size ---
    const size_t result_size = slice_len0 * slice_len1 * slice_len2;
    std::vector<T> result_1d(result_size);

    // --- Calculate Strides (Row-Major Order) ---
    const size_t stride_dim2 = 1;
    const size_t stride_dim1 = dim2 * stride_dim2;
    const size_t stride_dim0 = dim1 * stride_dim1;

    // --- Optimization Strategy ---

    // 1. Slice only in the last dimension (dim2)
    // Pattern: data[:, :, start2:stop2]
    if (slice_len0 == dim0 && slice_len1 == dim1) {
        const size_t elements_to_copy_per_row = slice_len2;
        size_t result_idx = 0;
        for (size_t n = 0; n < dim0; ++n) {
            for (size_t c = 0; c < dim1; ++c) {
                const size_t src_start_idx = n * stride_dim0 + c * stride_dim1 + final_start2;
                std::copy(data_1d.data() + src_start_idx,
                          data_1d.data() + src_start_idx + elements_to_copy_per_row,
                          result_1d.data() + result_idx);
                result_idx += elements_to_copy_per_row;
            }
        }
        return result_1d;
    }

    // 2. Slice only in the middle dimension (dim1) and full last dimension (dim2)
    // Pattern: data[:, start1:stop1, :] (when start2==0 && stop2==dim2 implicitly)
    if (slice_len0 == dim0 && slice_len2 == dim2) {
        // Check if the slice on dim2 is indeed the full dimension
        if (final_start2 == 0 && final_stop2 == dim2) {
             const size_t elements_to_copy_per_n_block = slice_len1 * dim2;
             size_t result_idx = 0;
             for (size_t n = 0; n < dim0; ++n) {
                 const size_t src_start_idx = n * stride_dim0 + final_start1 * stride_dim1;
                 std::copy(data_1d.data() + src_start_idx,
                           data_1d.data() + src_start_idx + elements_to_copy_per_n_block,
                           result_1d.data() + result_idx);
                 result_idx += elements_to_copy_per_n_block;
             }
             return result_1d;
        }
        // If slicing dim2 too in a non-optimized way, fall through to general case.
    }

    // 3. Slice only in the first dimension (dim0) and full middle/last dimensions
    // Pattern: data[start0:stop0, :, :] (when start1==0, stop1==dim1, start2==0, stop2==dim2 implicitly)
    if (slice_len1 == dim1 && slice_len2 == dim2) {
        // Check if the slices on dim1 and dim2 are indeed the full dimensions
        if (final_start1 == 0 && final_stop1 == dim1 &&
            final_start2 == 0 && final_stop2 == dim2) {
             const size_t elements_to_copy = slice_len0 * dim1 * dim2;
             const size_t src_start_idx = final_start0 * stride_dim0;
             std::copy(data_1d.data() + src_start_idx,
                       data_1d.data() + src_start_idx + elements_to_copy,
                       result_1d.data());
             return result_1d;
        }
        // If slicing dim1 or dim2 too, fall through to general case.
    }

    // --- General Case (Fallback for complex or non-contiguous slices) ---
    // This handles all other cases.
    size_t result_idx = 0;
    for (size_t n = final_start0; n < final_stop0; ++n) {
        const size_t base_offset_n = n * stride_dim0;
        for (size_t c = final_start1; c < final_stop1; ++c) {
            const size_t base_offset_nc = base_offset_n + c * stride_dim1;
            for (size_t l = final_start2; l < final_stop2; ++l) {
                const size_t original_index = base_offset_nc + l * stride_dim2;
                result_1d[result_idx++] = data_1d[original_index];
            }
        }
    }

    return result_1d;
}

// --- Convenience Wrappers ---
// These wrappers now pass dimX instead of -1 to signify "the end of the dimension".

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_from(std::vector<T>& data_1d,
                       size_t dim0, size_t dim1, size_t dim2,
                       int start2) {
    // Slice all of dim0 and dim1, and from start2 to the end of dim2.
    // Pass dim2 as stop2 to indicate the end.
    return slice_3d_optimized(data_1d, dim0, dim1, dim2,
                              0, static_cast<int>(dim0), // [:] for dim0
                              0, static_cast<int>(dim1), // [:] for dim1
                              start2, static_cast<int>(dim2)); // [start2:] for dim2
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_last_n(std::vector<T>& data_1d,
                         size_t dim0, size_t dim1, size_t dim2,
                         size_t n) {
    // Calculate the correct start index for the last N elements.
    // Handle case where n >= dim2.
    int start2 = (n >= dim2) ? 0 : static_cast<int>(dim2 - n);
    // Slice all of dim0 and dim1, and from start2 to the end of dim2.
    // Pass dim2 as stop2 to indicate the end.
    return slice_3d_optimized(data_1d, dim0, dim1, dim2,
                              0, static_cast<int>(dim0), // [:] for dim0
                              0, static_cast<int>(dim1), // [:] for dim1
                              start2, static_cast<int>(dim2)); // [-n:] -> [dim2-n:] for dim2
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_range(std::vector<T>& data_1d,
                        size_t dim0, size_t dim1, size_t dim2,
                        int start2, int stop2) {
    // Slice all of dim0 and dim1, and the specified range in dim2.
    return slice_3d_optimized(data_1d, dim0, dim1, dim2,
                              0, static_cast<int>(dim0), // [:] for dim0
                              0, static_cast<int>(dim1), // [:] for dim1
                              start2, stop2); // [start2:stop2] for dim2
}

// --- Utility to save vector to file ---
template<typename T>
void save_vector_to_file(const std::vector<T>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    file << std::fixed << std::setprecision(10); // Set precision for floats
    for (size_t i = 0; i < vec.size(); ++i) {
        file << vec[i];
        if (i < vec.size() - 1) {
            file << "\n"; // Newline separated
        }
    }
    file.close();
}

// --- Utility to load vector from file ---
template<typename T>
std::vector<T> load_vector_from_file(const std::string& filename) {
     std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    std::vector<T> vec;
    std::string line;
    while(std::getline(file, line)) {
        if (!line.empty()) {
             std::stringstream ss(line);
             T value;
             ss >> value;
             vec.push_back(value);
        }
    }
    file.close();
    return vec;
}


#endif // SLICE_3D_TPP