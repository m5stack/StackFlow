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

#include "middleware/axcl_base.hpp"

namespace middleware {

#if defined(CONFIG_AXCL_RUNTIME_API_ENABLE)
class runtime_runner final : public axcl_base {
public:
    runtime_runner();
    ~runtime_runner() override;

    [[nodiscard]] bool init(const std::string& config_file, const uint32_t& device_index, const uint32_t& kind) override;
    [[nodiscard]] bool final() override;

    [[nodiscard]] bool load(const std::string& model_path) override;
    [[nodiscard]] bool prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) override;

    [[nodiscard]] bool run(const bool& parallel) override;
    [[nodiscard]] bool set() override;

    [[nodiscard]] uint32_t get_input_count() const override;
    [[nodiscard]] uint32_t get_output_count() const override;
    [[nodiscard]] std::string get_input_name(const uint32_t& index) const override;
    [[nodiscard]] std::string get_output_name(const uint32_t& index) const override;
    [[nodiscard]] void *get_input_pointer(const uint32_t& index) const override;
    [[nodiscard]] void *get_output_pointer(const uint32_t& index) const override;
    [[nodiscard]] uintmax_t get_input_size(const uint32_t& index) const override;
    [[nodiscard]] uintmax_t get_output_size(const uint32_t& index) const override;
    [[nodiscard]] uintmax_t get_shape_group_count() const override;

    [[nodiscard]] std::string get_library_version() const override;
    [[nodiscard]] std::string get_model_version() const override;

    [[nodiscard]] int32_t get_model_type() const override;
    [[nodiscard]] int32_t get_npu_type() const override;
    [[nodiscard]] int32_t get_batch_size() const override;
    [[nodiscard]] intmax_t get_sys_usage() const override;
    [[nodiscard]] intmax_t get_cmm_usage() const override;

private:
    struct impl;
    impl *impl_;
};
#endif

}
