/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include "middleware/runner.hpp"

#include "utilities/file.hpp"
#include "utilities/log.hpp"

#include <chrono>
#include <thread>

void middleware::runner::sleep_for(const uint32_t sleep_duration) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration));
}

std::vector<std::string> middleware::runner::read(const std::string& list_file) {
    if (!utilities::exists(list_file) || !utilities::is_regular_file(list_file)) {
        utilities::glog.print(utilities::log::type::error,
            "Stimulus list file{%s} is not exist.\n", list_file.c_str());
        return {};
    }

    std::vector<std::string> list;

    std::ifstream list_stream;
    list_stream.open(list_file, std::ios::in);
    if (!list_stream.is_open()) {
        utilities::glog.print(utilities::log::type::error,
            "Read stimulus list file{%s} failed.\n", list_file.c_str());
        return {};
    }

    std::string file_path;
    while (std::getline(list_stream, file_path)) {
        list.push_back(file_path);
    }
    list_stream.close();

    return list;
}

bool middleware::runner::read(const std::string& binary_file, void* address, const uintmax_t& size) {
    if (!utilities::exists(binary_file) || !utilities::is_regular_file(binary_file)) {
        utilities::glog.print(utilities::log::type::error,
            "Stimulus file{%s} is not exist.\n", binary_file.c_str());
        return false;
    }

    const auto file_size = utilities::file_size(binary_file);
    if (utilities::error_size == file_size) {
        utilities::glog.print(utilities::log::type::error, "Get stimulus file size failed.\n");
        return false;
    }

    if (file_size != size) {
        utilities::glog.print(utilities::log::type::error,
            "Size of stimulus file{%s} is not equal the specified{file: %d vs tensor: %d}.\n", binary_file.c_str(), file_size, size);
        return false;
    }

    return utilities::read(binary_file, address, size);
}

bool middleware::runner::write(const std::string& binary_file, const void* address, const uintmax_t& size) {
    return utilities::write(binary_file, address, size);
}

bool middleware::runner::verify(const void* dst, const void* src, const uintmax_t& size) {
    const auto* src_ptr = static_cast<const uint8_t*>(src);
    const auto* dst_ptr = static_cast<const uint8_t*>(dst);

    for (uintmax_t i = 0; i < size; i++) {
        if (src_ptr[i] != dst_ptr[i]) {
            utilities::glog.print(utilities::log::type::error,
                "Verify stimulus failed{offset: 0x%x; 0x%02X vs. 0x%02X}.\n", i, src_ptr[i], dst_ptr[i]);
            return false;
        }
    }

    return true;
}
