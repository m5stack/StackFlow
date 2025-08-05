/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace middleware {

class runner {
public:
    runner() = default;
    virtual ~runner() = default;

    [[nodiscard]] virtual bool init(const std::string& config_file, const uint32_t& device_index, const uint32_t& kind) = 0;
    [[nodiscard]] virtual bool final() = 0;

    [[nodiscard]] virtual bool load(const std::string& model_path) = 0;
    [[nodiscard]] virtual bool prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) = 0;

    [[nodiscard]] virtual bool run(const bool& parallel) = 0;

    [[nodiscard]] virtual uint32_t get_input_count() const = 0;
    [[nodiscard]] virtual uint32_t get_output_count() const = 0;
    [[nodiscard]] virtual std::string get_input_name(const uint32_t& index) const = 0;
    [[nodiscard]] virtual std::string get_output_name(const uint32_t& index) const = 0;
    [[nodiscard]] virtual void *get_input_pointer(const uint32_t& index) const = 0;
    [[nodiscard]] virtual void *get_output_pointer(const uint32_t& index) const = 0;
    [[nodiscard]] virtual uintmax_t get_input_size(const uint32_t& index) const = 0;
    [[nodiscard]] virtual uintmax_t get_output_size(const uint32_t& index) const = 0;

    [[nodiscard]] virtual uintmax_t get_shape_group_count() const = 0;

    [[nodiscard]] virtual bool flush_input() const = 0;
    [[nodiscard]] virtual bool invalidate_output() const = 0;

    [[nodiscard]] virtual bool feed(const std::string& input_folder, const std::string& stimulus_name) const = 0;
    [[nodiscard]] virtual bool verify(const std::string& output_folder, const std::string& stimulus_name) const = 0;
    [[nodiscard]] virtual bool save(const std::string& output_folder, const std::string& stimulus_name) const = 0;

    [[nodiscard]] virtual std::string get_library_version() const = 0;
    [[nodiscard]] virtual std::string get_model_version() const = 0;

    [[nodiscard]] virtual int32_t get_model_type() const = 0;
    [[nodiscard]] virtual int32_t get_npu_type() const = 0;
    [[nodiscard]] virtual int32_t get_batch_size() const = 0;
    [[nodiscard]] virtual intmax_t get_sys_usage() const = 0;
    [[nodiscard]] virtual intmax_t get_cmm_usage() const = 0;

    virtual void sleep_for(uint32_t sleep_duration) const;

    [[nodiscard]] static std::vector<std::string> read(const std::string& list_file);
    [[nodiscard]] static bool read(const std::string& binary_file, void* address, const uintmax_t& size);
    [[nodiscard]] static bool write(const std::string& binary_file, const void* address, const uintmax_t& size);
    [[nodiscard]] static bool verify(const void* dst, const void* src, const uintmax_t& size);
};

}
