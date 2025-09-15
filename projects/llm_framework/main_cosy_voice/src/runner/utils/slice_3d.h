// src/slice_3d.h
#ifndef SLICE_3D_H
#define SLICE_3D_H

#include <vector>
#include <cstddef>
#include <type_traits>

// Template declarations
template <typename T>
typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, size_t>::type
normalize_slice_index(T index, size_t dim_size);

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_optimized(std::vector<T>& data_1d,
                   size_t dim0, size_t dim1, size_t dim2,
                   int start0, int stop0,
                   int start1, int stop1,
                   int start2, int stop2);

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_from(std::vector<T>& data_1d,
                       size_t dim0, size_t dim1, size_t dim2,
                       int start2);

// 这个函数有bug
// template <typename T>
// typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
// slice_3d_last_dim_last_n(std::vector<T>& data_1d,
//                          size_t dim0, size_t dim1, size_t dim2,
//                          size_t n);

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_range(std::vector<T>& data_1d,
                        size_t dim0, size_t dim1, size_t dim2,
                        int start2, int stop2);

// Include the implementation for templates
#include "slice_3d.tpp"

#endif // SLICE_3D_H