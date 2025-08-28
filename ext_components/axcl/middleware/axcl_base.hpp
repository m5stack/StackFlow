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
#include <global_config.h>

#include "middleware/runner.hpp"

#if defined(CONFIG_AXCL_RUNTIME_API_ENABLE) || defined(CONFIG_AXCL_NATIVE_API_ENABLE)

#include <functional>

namespace middleware {

class axcl_base : public runner {
public:
    using npu_func = std::function<bool()>;

    [[nodiscard]] static bool init(const std::string& config, const uint32_t& index, const npu_func& func);
    [[nodiscard]] static bool final(const npu_func& func);
    [[nodiscard]] static bool set_context();

    [[nodiscard]] bool flush_input() const override;
    [[nodiscard]] bool invalidate_output() const override;

    [[nodiscard]] bool feed(const std::string& input_folder, const std::string& stimulus_name) const override;
    [[nodiscard]] bool verify(const std::string& output_folder, const std::string& stimulus_name) const override;
    [[nodiscard]] bool save(const std::string& output_folder, const std::string& stimulus_name) const override;

private:
    [[nodiscard]] static bool feed(const std::string& folder, const std::string& stimulus_name, const std::string& tensor_name, void* address, uintmax_t size) ;
    [[nodiscard]] static bool verify(const std::string &folder, const std::string &stimulus_name, const std::string &tensor_name, void *address, uintmax_t size) ;
    [[nodiscard]] static bool save(const std::string& output_folder, const std::string& stimulus_name, const std::string& tensor_name, void* address, uintmax_t size);
};

}
#endif
