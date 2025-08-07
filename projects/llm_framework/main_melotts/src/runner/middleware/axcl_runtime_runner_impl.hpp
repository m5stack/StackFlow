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

#include "middleware/axcl_runtime_runner.hpp"

#include "middleware/axcl_base.hpp"
#include "utilities/scalar_guard.hpp"
#include "utilities/file.hpp"
#include "utilities/log.hpp"

#include <axcl.h>

#if defined(ENV_AXCL_RUNTIME_API_ENABLE)
struct middleware::runtime_runner::impl {
    impl() = default;

    ~impl() {
        std::ignore = this->final();
    }

    [[nodiscard]] bool init(const std::string& config_file, const uint32_t& index, const uint32_t& kind) {
        // 0. check the NPU kind
        if (kind > AXCL_VNPU_LITTLE_BIG) {
            utilities::glog.print(utilities::log::type::error,
                "Specified NPU kind{%d} is out of range{total %d}.\n", kind, static_cast<int>(AXCL_VNPU_LITTLE_BIG) + 1);
            return false;
        }

        // 1. build the npu init function
        auto npu_init_func = [&kind]() {
            if (const int ret = axclrtEngineInit(static_cast<axclrtEngineVNpuKind>(kind)); 0 != ret) {
                utilities::glog.print(utilities::log::type::error, "Init AXCLRT Engine as kind{%d} failed{0x%08X}.\n", kind, ret);
                return false;
            }
            utilities::glog.print(utilities::log::type::info, "AXCLRT Engine inited.\n");
            return true;
        };

        // 2. execute initialization
        if (const auto ret = axcl_base::init(config_file, index, npu_init_func); !ret) {
            return false;
        }

        // 3. set the flag
        this->is_initialized_ = true;
        return true;
    }

    [[nodiscard]] bool final() {
        auto flag = true;

        if (this->is_initialized_) {
            if (0 != this->model_id_) {
                for (auto& input : inputs_) {
                    if (nullptr != input) {
                        std::ignore = axclrtFree(input);
                        input = nullptr;
                    }
                }
                for (auto& output : outputs_) {
                    if (nullptr != output) {
                        std::ignore = axclrtFree(output);
                        output = nullptr;
                    }
                }

                flag &= 0 == axclrtEngineDestroyIOInfo(this->info_);
                flag &= 0 == axclrtEngineDestroyIO(this->io_);
                flag &= 0 == axclrtEngineUnload(this->model_id_);
            }

             flag &= 0 == axcl_base::final([&]() {
                if (const auto ret = axclrtEngineFinalize(); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Final AXCLRT Engine failed{0x%08X}.\n", ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "AXCLRT Engine finalized.\n");
                return true;
            });

            this->is_initialized_ = false;
        }

        return flag;
    }

    [[nodiscard]] bool load(const std::string& model_path) {
        if (!this->is_initialized_) {
            utilities::glog.print(utilities::log::type::error, "axcl is not initialized.\n");
            return false;
        }

        if (0 != this->model_id_) {
            utilities::glog.print(utilities::log::type::error, "Model is already loaded.\n");
            return false;
        }

        if (!utilities::exists(model_path)
            || !utilities::is_regular_file(model_path)
            || 0 == utilities::file_size(model_path)
            || utilities::error_size == utilities::file_size(model_path)) {
            utilities::glog.print(utilities::log::type::error, "Model file{%s} error, please check it.\n", model_path.c_str());
            return false;
        }

        if (const auto ret = axclrtEngineLoadFromFile(model_path.c_str(), &this->model_id_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Create model{%s} handle failed.\n", model_path.c_str());
            return false;
        }

        if (const auto ret = axclrtEngineCreateContext(this->model_id_, &this->context_id_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Create model{%s} context failed.\n", model_path.c_str());
            return false;
        }

        utilities::glog.print(utilities::log::type::info, "Model ID: %u, Context ID: %u\n", 
                         this->model_id_, this->context_id_);

        return true;
    }

    [[nodiscard]] bool prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) {
        std::ignore = input_cached;
        std::ignore = output_cached;

        // 0. check the handle
        if (0 == this->model_id_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return false;
        }

        // 1. get the IO info
        if (const auto ret = axclrtEngineGetIOInfo(this->model_id_, &this->info_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model io info failed{0x%08X}.\n", ret);
            return false;
        }

        // 2. get the count of shape group
        int32_t total_group = 0;
        if (const auto ret = axclrtEngineGetShapeGroupsCount(this->info_, &total_group); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model shape group count failed{0x%08X}.\n", ret);
            return false;
        }

        // 3. check the group index
        if (group >= static_cast<decltype(group)>(total_group)) {
            utilities::glog.print(utilities::log::type::error, "Model group{%d} is out of range{total %d}.\n", group, total_group);
            return false;
        }
        this->group_ = static_cast<int32_t>(group);

        // 4. check the batch size
        this->batch_ = (0 == batch ? 1 : batch);

        // 5. get the count of inputs
        uint32_t input_count = 0;
        if (input_count = axclrtEngineGetNumInputs(this->info_); 0 == input_count) {
            utilities::glog.print(utilities::log::type::error, "Get model input count failed.\n");
            return false;
        }

        // 6. get the count of outputs
        uint32_t output_count = 0;
        if (output_count = axclrtEngineGetNumOutputs(this->info_); 0 == output_count) {
            utilities::glog.print(utilities::log::type::error, "Get model output count failed.\n");
            return false;
        }

        // 7. prepare the input and output
        this->inputs_.resize(input_count, nullptr);
        this->inputs_size_.resize(input_count, 0);
        this->outputs_.resize(output_count, nullptr);
        this->outputs_size_.resize(output_count, 0);

        // 8. prepare the memory, inputs
        for (uint32_t i = 0; i < input_count; i++) {
            uint32_t original_size = 0;
            if (original_size = axclrtEngineGetInputSizeByIndex(this->info_, group, i); 0 == original_size) {
                utilities::glog.print(utilities::log::type::error, "Get model input{index: %d} size failed.\n", i);
                return false;
            }

            this->inputs_size_[i] = original_size * this->batch_;

            if (const auto ret = axclrtMalloc(&this->inputs_[i], this->inputs_size_[i], axclrtMemMallocPolicy{}); 0 != ret) {
                utilities::glog.print(utilities::log::type::error, "Memory allocation for tensor{index: %d} failed{0x%08X}.\n", i, ret);
                return false;
            }
            utilities::glog.print(utilities::log::type::info, "Memory for tensor{index: %d} is allocated.\n", i);

            // clean memory, some cases model may need to clean memory
            // axclrtMemset(this->inputs_[i], 0, size);

            utilities::glog.print(utilities::log::type::info,
                "Set tensor { index: %d, address: %p, size: %u Bytes }.\n", i, this->inputs_[i], this->inputs_size_[i]);
        }

        // 9. prepare the memory, outputs
        for (uint32_t i = 0; i < output_count; i++) {
            uint32_t original_size = 0;
            if (original_size = axclrtEngineGetOutputSizeByIndex(this->info_, group, i); 0 == original_size) {
                utilities::glog.print(utilities::log::type::error, "Get model output{index: %d} size failed.\n", i);
                return false;
            }

            this->outputs_size_[i] = original_size * this->batch_;

            if (const auto ret = axclrtMalloc(&this->outputs_[i], this->outputs_size_[i], axclrtMemMallocPolicy{}); 0 != ret) {
                utilities::glog.print(utilities::log::type::error, "Memory allocation for tensor{index: %d} failed{0x%08X}.\n", i, ret);
                return false;
            }
            utilities::glog.print(utilities::log::type::info, "Memory for tensor{index: %d} is allocated.\n", i);

            // clean memory, some cases model may need to clean memory
            // axclrtMemset(this->outputs_[i], 0, size);

            utilities::glog.print(utilities::log::type::info,
                "Set tensor { index: %d, address: %p, size: %u Bytes }.\n", i, this->outputs_[i], this->outputs_size_[i]);
        }

        // 10. create the IO
        if (const auto ret = axclrtEngineCreateIO(this->info_, &this->io_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Create model io failed{0x%08X}.\n", ret);
            return false;
        }
        utilities::glog.print(utilities::log::type::info, "AXCLRT Engine inited.\n");

        // 11. set the input and output buffer
        for (uint32_t i = 0; i < input_count; i++) {
            if (const auto ret = axclrtEngineSetInputBufferByIndex(this->io_, i, this->inputs_[i], this->inputs_size_[i]); 0 != ret) {
                utilities::glog.print(utilities::log::type::error, "Set input buffer{index: %d} failed{0x%08X}.\n", i, ret);
                return false;
            }
        }
        for (uint32_t i = 0; i < output_count; i++) {
            if (const auto ret = axclrtEngineSetOutputBufferByIndex(this->io_, i, this->outputs_[i], this->outputs_size_[i]); 0 != ret) {
                utilities::glog.print(utilities::log::type::error, "Set output buffer{index: %d} failed{0x%08X}.\n", i, ret);
                return false;
            }
        }

        // 12. set the batch size
        if (const auto ret = axclrtEngineSetDynamicBatchSize(this->io_, this->batch_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Set batch size{%d} failed{0x%08X}.\n", this->batch_, ret);
            return false;
        }

        return true;
    }

    [[nodiscard]] bool set() {
        if (const auto ret = axcl_base::set_context(); !ret) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool run(const bool& parallel) const {
        std::ignore = parallel;

        if (0 == this->model_id_ || 0 == this->context_id_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return false;
        }

        if (nullptr == this->io_) {
            utilities::glog.print(utilities::log::type::error, "Model io is not set, prepare first.\n");
            return false;
        }

        if (const auto ret = axclrtEngineExecute(this->model_id_, this->context_id_, this->group_, this->io_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Run model failed{0x%08X}.\n", ret);
            return false;
        }
        utilities::glog.print(utilities::log::type::info, "Running done.\n");

        return true;
    }

    [[nodiscard]] uint32_t get_input_count() const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        return axclrtEngineGetNumInputs(this->info_);
    }

    [[nodiscard]] uint32_t get_output_count() const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        return axclrtEngineGetNumOutputs(this->info_);
    }

    [[nodiscard]] std::string get_input_name(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return {};
        }

        if (index >= get_input_count()) {
            return {};
        }
        return std::string{axclrtEngineGetInputNameByIndex(this->info_, index)};
    }

    [[nodiscard]] std::string get_output_name(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return {};
        }

        if (index >= get_output_count()) {
            return {};
        }
        return std::string{axclrtEngineGetOutputNameByIndex(this->info_, index)};
    }

    [[nodiscard]] void *get_input_pointer(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return nullptr;
        }

        if (index >= get_input_count()) {
            return nullptr;
        }
        return this->inputs_[index];
    }

    [[nodiscard]] void *get_output_pointer(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return nullptr;
        }

        if (index >= get_output_count()) {
            return nullptr;
        }
        return this->outputs_[index];
    }

    [[nodiscard]] uintmax_t get_input_size(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        if (index >= get_input_count()) {
            return 0;
        }
        return axclrtEngineGetInputSizeByIndex(this->info_, this->group_, index);
    }

    [[nodiscard]] uintmax_t get_output_size(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        if (index >= get_output_count()) {
            return 0;
        }
        return axclrtEngineGetOutputSizeByIndex(this->info_, this->group_, index);
    }

    [[nodiscard]] uintmax_t get_shape_group_count() const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        int group = 0;
        if (const auto ret = axclrtEngineGetShapeGroupsCount(this->info_, &group); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model shape group count failed{0x%08X}.\n", ret);
            return 0;
        }
        return group;
    }

    [[nodiscard]] static std::string get_library_version() {
        int32_t major = 0, minor = 0, patch = 0;
        if (const auto ret = axclrtGetVersion(&major, &minor, &patch); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get library version failed{0x%08X}.\n", ret);
            return "";
        }
        return {std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch)};
    }

    [[nodiscard]] std::string get_model_version() const {
        if (0 == this->model_id_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return "";
        }
        return {axclrtEngineGetModelCompilerVersion(this->model_id_)};
    }

    [[nodiscard]] int32_t get_model_type() const {
        if (0 == this->model_id_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return -1;
        }

        axclrtEngineModelKind type;
        if (const auto ret = axclrtEngineGetModelTypeFromModelId(this->model_id_, &type); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model type failed{0x%08X}.\n", ret);
            return -1;
        }
        return type;
    }

    [[nodiscard]] int32_t get_npu_type() const {
        if (!this->is_initialized_) {
            utilities::glog.print(utilities::log::type::error, "axcl is not initialized.\n");
            return false;
        }

        axclrtEngineVNpuKind kind;
        if (const auto ret = axclrtEngineGetVNpuKind(&kind); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get visual NPU type failed{0x%08X}.\n", ret);
            return -1;
        }
        return kind;
    }

    [[nodiscard]] int32_t get_batch_size() const {
        if (0 == this->model_id_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return -1;
        }

        return static_cast<int32_t>(this->batch_);
    }

    [[nodiscard]] intmax_t get_sys_usage() const {
        if (0 == this->model_id_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return -1;
        }

        int64_t sys_size = 0, cmm_size = 0;
        if (const auto ret = axclrtEngineGetUsageFromModelId(this->model_id_, &sys_size, &cmm_size); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model usage failed{0x%08X}.\n", ret);
            return -1;
        }

        return sys_size;
    }

    [[nodiscard]] intmax_t get_cmm_usage() const {
        if (0 == this->model_id_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return -1;
        }

        int64_t sys_size = 0, cmm_size = 0;
        if (const auto ret = axclrtEngineGetUsageFromModelId(this->model_id_, &sys_size, &cmm_size); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model usage failed{0x%08X}.\n", ret);
            return -1;
        }

        return cmm_size;
    }

private:
    int32_t group_ = 0;
    uint32_t batch_ = 0;

    uint64_t model_id_{};
    uint64_t context_id_{};
    axclrtEngineIOInfo info_{};
    axclrtEngineIO io_{};
    std::vector<void*> inputs_;
    std::vector<void*> outputs_;
    std::vector<uintmax_t> inputs_size_;
    std::vector<uintmax_t> outputs_size_;

    bool is_initialized_ = false;
};
#endif
