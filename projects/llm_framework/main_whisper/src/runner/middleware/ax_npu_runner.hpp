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

#if defined(ENV_AX_LOCAL_API_ENABLE)
#include "middleware/runner.hpp"

namespace middleware {

class npu_runner final : public runner {
public:
    npu_runner();
    ~npu_runner() override;

    [[nodiscard]] bool init(const std::string& config_file, const uint32_t& index, const uint32_t& kind) override;
    [[nodiscard]] bool final() override;

    [[nodiscard]] bool load(const std::string& model_path) override;
    [[nodiscard]] bool prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) override;

    [[nodiscard]] bool run(const bool& parallel) override;

    [[nodiscard]] uint32_t get_input_count() const override;
    [[nodiscard]] uint32_t get_output_count() const override;
    [[nodiscard]] std::string get_input_name(const uint32_t& index) const override;
    [[nodiscard]] std::string get_output_name(const uint32_t& index) const override;
    [[nodiscard]] void *get_input_pointer(const uint32_t& index) const override;
    [[nodiscard]] void *get_output_pointer(const uint32_t& index) const override;
    [[nodiscard]] uintmax_t get_input_size(const uint32_t& index) const override;
    [[nodiscard]] uintmax_t get_output_size(const uint32_t& index) const override;
    [[nodiscard]] uintmax_t get_shape_group_count() const override;

    [[nodiscard]] bool flush_input() const override;
    [[nodiscard]] bool invalidate_output() const override;

    [[nodiscard]] bool feed(const std::string& input_folder, const std::string& stimulus_name) const override;
    [[nodiscard]] bool verify(const std::string& output_folder, const std::string& stimulus_name) const override;
    [[nodiscard]] bool save(const std::string& output_folder, const std::string& stimulus_name) const override;

    [[nodiscard]] std::string get_library_version() const override;
    [[nodiscard]] std::string get_model_version() const override;

    [[nodiscard]] int32_t get_model_type() const override;
    [[nodiscard]] int32_t get_npu_type() const override;
    [[nodiscard]] int32_t get_batch_size() const override;
    [[nodiscard]] intmax_t get_sys_usage() const override;
    [[nodiscard]] intmax_t get_cmm_usage() const override;

    void sleep_for(uint32_t sleep_duration) const override;

private:
    struct impl;
    impl *impl_;
};
#endif

}
