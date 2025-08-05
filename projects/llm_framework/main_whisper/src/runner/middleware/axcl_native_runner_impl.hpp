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

#include "middleware/axcl_native_runner.hpp"

#include "utilities/scalar_guard.hpp"
#include "utilities/file.hpp"
#include "utilities/file_mapper.hpp"
#include "utilities/log.hpp"

#include <axcl.h>

#if defined(ENV_AXCL_NATIVE_API_ENABLE)
#define MODEL_LEADING_NAME "axcl npu "

struct middleware::native_runner::impl {
    impl() = default;

    ~impl() {
        std::ignore = this->final();
    }

    [[nodiscard]] bool init(const std::string& config_file, const uint32_t& index, const uint32_t& kind) {
        // 0. check the NPU kind
        if (kind >= ::AX_ENGINE_VIRTUAL_NPU_BUTT) {
            utilities::glog.print(utilities::log::type::error,
                "Specified NPU kind{%d} is out of range{total %d}.\n", kind, ::AX_ENGINE_VIRTUAL_NPU_BUTT);
            return false;
        }

        // 1. build the npu init function
        auto npu_init_func = [&kind]() {
            ::AX_ENGINE_NPU_ATTR_T attr{};
            attr.eHardMode = static_cast<::AX_ENGINE_NPU_MODE_T>(kind);
            if (const int ret = ::AXCL_ENGINE_Init(&attr); 0 != ret) {
                utilities::glog.print(utilities::log::type::error,
                    "Init axcl NPU as kind{%d} failed{0x%08X}.\n", kind, ret);
                return false;
            }
            utilities::glog.print(utilities::log::type::info, "axcl NPU inited.\n");
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
            if (nullptr != this->handle_) {
                for (auto& input : inputs_) {
                    if (0 != input.phyAddr) {
                        flag &= 0 == ::axclrtFree(reinterpret_cast<void *>(input.phyAddr));
                        input.phyAddr = 0;
                    }
                }
                for (auto& output : outputs_) {
                    if (0 != output.phyAddr) {
                        flag &= 0 == ::axclrtFree(reinterpret_cast<void *>(output.phyAddr));
                        output.phyAddr = 0;
                    }
                }

                flag &= 0 == ::AXCL_ENGINE_DestroyHandle(this->handle_);
            }

            flag &= 0 == axcl_base::final([&]() {
               if (const auto ret = ::AXCL_ENGINE_Deinit(); 0 != ret) {
                   utilities::glog.print(utilities::log::type::error, "Final axcl NPU failed{0x%08X}.\n", ret);
                   return false;
               }
               utilities::glog.print(utilities::log::type::info, "axcl NPU finalized.\n");
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

        if (nullptr != this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model is already loaded.\n");
            return false;
        }

        if (!utilities::exists(model_path)
            || !utilities::is_regular_file(model_path)
            || 0 == utilities::file_size(model_path)
            || utilities::error_size == utilities::file_size(model_path)) {
            utilities::glog.print(utilities::log::type::error,
                "Model file{%s} error, please check it.\n", model_path.c_str());
            return false;
        }

        const auto file = utilities::file_mapper(model_path);
        if (nullptr == file.get()) {
            utilities::glog.print(utilities::log::type::error,
                "Model file{%s} mapping failed.\n", model_path.c_str());
            return false;
        }

        const auto file_size = utilities::file_size(model_path);
        auto device = utilities::scalar_guard<void*>(
            [&file_size]() -> void* {
                void *mem = nullptr;
                if (const auto ret = ::axclrtMalloc(&mem, file_size, ::axclrtMemMallocPolicy{}); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error,
                        "Malloc on device for model file failed{0x%08X}.\n", ret);
                }
                return mem;
            },
            [](void*& dev) { if (dev != nullptr) std::ignore = ::axclrtFree(dev); }
        );

        if (nullptr == device.get()) {
            utilities::glog.print(utilities::log::type::error,
                "Malloc on device for file{%s} failed.\n", model_path.c_str());
            return false;
        }

        if (const auto ret = ::axclrtMemcpy(device.get(), file.get(), file_size, ::AXCL_MEMCPY_HOST_TO_DEVICE); 0 != ret) {
            utilities::glog.print(utilities::log::type::error,
                "Send model file{%s} to device failed{0x%08X}.\n", model_path.c_str(), ret);
            return false;
        }

        const auto model_name = std::string(MODEL_LEADING_NAME) + utilities::get_file_name(model_path);
        ::AX_ENGINE_HANDLE_EXTRA_T extra_param{};
        extra_param.pName = reinterpret_cast<::AX_S8 *>(const_cast<char *>(model_name.c_str()));
        if (const auto ret = ::AXCL_ENGINE_CreateHandleV2(&this->handle_, device.get(), file_size, &extra_param); 0 != ret) {
            utilities::glog.print(utilities::log::type::error,
                "Create model{%s} handle failed.\n", model_path.c_str());
            return false;
        }

        if (const auto ret = ::AXCL_ENGINE_CreateContextV2(this->handle_, &this->context_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error,
                "Create model{%s} context failed.\n", model_path.c_str());
            return false;
        }

        return true;
    }

    [[nodiscard]] bool prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) {
        std::ignore = input_cached;
        std::ignore = output_cached;

        // 0. check the handle
        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is null.\n");
            return false;
        }

        // 1. get the count of shape group
        if (const auto ret = ::AXCL_ENGINE_GetGroupIOInfoCount(this->handle_, &this->group_count_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error,
                "Get model shape group count failed{0x%08X}.\n", ret);
            return false;
        }

        // 2. check the group index
        if (group >= this->group_count_) {
            utilities::glog.print(utilities::log::type::error,
                "Model group{%d} is out of range{total %d}.\n", group, this->group_);
            return false;
        }
        this->group_ = group;

        // 3. get the IO info
        if (1 == this->group_count_) {
            if (const auto ret = ::AXCL_ENGINE_GetIOInfo(this->handle_, &info_); 0 != ret) {
                utilities::glog.print(utilities::log::type::error,
                    "Get model IO info failed{0x%08X}.\n", ret);
                return false;
            }
        } else {
            if (const auto ret = ::AXCL_ENGINE_GetGroupIOInfo(this->handle_, group, &info_); 0 != ret) {
                utilities::glog.print(utilities::log::type::error,
                    "Get model group{index: %d} IO info failed{0x%08X}.\n", group, ret);
                return false;
            }
        }

        // 4. check the batch size
        this->batch_ = (0 == batch ? 1 : batch);
        this->io_.nBatchSize = this->batch_;

        // 5. prepare the input and output
        this->inputs_.resize(this->info_->nInputSize, ::AX_ENGINE_IO_BUFFER_T{});
        this->io_.pInputs = this->inputs_.data();
        this->io_.nInputSize = this->info_->nInputSize;

        this->outputs_.resize(this->info_->nOutputSize, ::AX_ENGINE_IO_BUFFER_T{});
        this->io_.pOutputs = this->outputs_.data();
        this->io_.nOutputSize = this->info_->nOutputSize;

        // 6. prepare the memory, inputs
        for (::AX_U32 i = 0; i < this->info_->nInputSize; i++) {
            const auto meta = &(this->info_->pInputs[i]);
            this->io_.pInputs[i].nSize = meta->nSize * this->batch_;

            if (const auto ret = ::axclrtMalloc(
                reinterpret_cast<void **>(&this->io_.pInputs[i].phyAddr),
                this->io_.pInputs[i].nSize,
                ::axclrtMemMallocPolicy{}); 0 != ret) {
                utilities::glog.print(utilities::log::type::error,
                    "Memory allocation for tensor {%s} failed{0x%08X}.\n", meta->pName, ret);
                return false;
            }
            utilities::glog.print(utilities::log::type::info, "Memory for tensor {%s} is allocated.\n", meta->pName);

            // clean memory, some cases model may need to clean memory
            //::axclrtMemset(reinterpret_cast<void *>(this->io_.pInputs[i].phyAddr), 0, this->io_.pInputs[i].nSize);

            utilities::glog.print(utilities::log::type::info,
                "Set tensor { name: %s, phy: %p, vir: %p, size: %u Bytes }.\n",
                meta->pName,
                reinterpret_cast<void*>(this->io_.pInputs[i].phyAddr),
                this->io_.pInputs[i].pVirAddr,
                this->io_.pInputs[i].nSize);
        }

        // 7. prepare the memory, outputs
        for (::AX_U32 i = 0; i < this->info_->nOutputSize; i++) {
            const auto meta = &(this->info_->pOutputs[i]);
            this->io_.pOutputs[i].nSize = meta->nSize * this->batch_;

            if (const auto ret = ::axclrtMalloc(
                reinterpret_cast<void **>(&this->io_.pOutputs[i].phyAddr),
                this->io_.pOutputs[i].nSize,
                ::axclrtMemMallocPolicy{}); 0 != ret) {
                utilities::glog.print(utilities::log::type::error,
                    "Memory allocation for tensor {%s} failed{0x%08X}.\n", meta->pName, ret);
                return false;
            }
            utilities::glog.print(utilities::log::type::info, "Memory for tensor {%s} is allocated.\n", meta->pName);

            // clean memory, some cases model may need to clean memory
            //::axclrtMemset(reinterpret_cast<void *>(this->io_.pOutputs[i].phyAddr), 0, this->io_.pOutputs[i].nSize);

            utilities::glog.print(utilities::log::type::info,
                "Set tensor { name: %s, phy: %p, vir: %p, size: %u Bytes }.\n",
                meta->pName,
                reinterpret_cast<void*>(this->io_.pOutputs[i].phyAddr),
                this->io_.pOutputs[i].pVirAddr,
                this->io_.pOutputs[i].nSize);
        }

        return true;
    }

    [[nodiscard]] bool run(const bool& parallel) {
        this->io_.nParallelRun = (parallel ? 1 : 0);

        if (nullptr == this->handle_ || nullptr == this->context_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return false;
        }

        if (0 == this->io_.nInputSize || 0 == this->io_.nOutputSize) {
            utilities::glog.print(utilities::log::type::error, "Model io is not set, prepare first.\n");
            return false;
        }

        ::AX_S32 ret = 0;
        // Note: this is a workaround for the case that the model has only ONE group,
        //       currently AXCL_ENGINE_GetGroupIOInfo will return error.
        //       the bug of libax_engine.so will be fixed in the future.
        if (1 == this->group_count_) {
            ret = ::AXCL_ENGINE_RunSyncV2(this->handle_, this->context_, &this->io_);
        } else {
            ret = ::AXCL_ENGINE_RunGroupIOSync(this->handle_, this->context_, this->group_, &this->io_);
        }
        if (0 != ret) {
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

        return this->info_->nInputSize;
    }

    [[nodiscard]] uint32_t get_output_count() const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        return this->info_->nOutputSize;
    }

    [[nodiscard]] std::string get_input_name(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return {};
        }

        if (index >= this->info_->nInputSize) {
            return {};
        }
        return std::string{this->info_->pInputs[index].pName};
    }

    [[nodiscard]] std::string get_output_name(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return {};
        }

        if (index >= this->info_->nOutputSize) {
            return {};
        }
        return std::string{this->info_->pOutputs[index].pName};
    }

    [[nodiscard]] void *get_input_pointer(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return nullptr;
        }

        if (index >= this->info_->nInputSize) {
            return nullptr;
        }
        return reinterpret_cast<void *>(this->io_.pInputs[index].phyAddr);
    }

    [[nodiscard]] void *get_output_pointer(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return nullptr;
        }

        if (index >= this->info_->nOutputSize) {
            return nullptr;
        }
        return reinterpret_cast<void *>(this->io_.pOutputs[index].phyAddr);
    }

    [[nodiscard]] uintmax_t get_input_size(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        if (index >= this->info_->nInputSize) {
            return 0;
        }
        return this->io_.pInputs[index].nSize;
    }

    [[nodiscard]] uintmax_t get_output_size(const uint32_t& index) const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        if (index >= this->info_->nOutputSize) {
            return 0;
        }
        return this->io_.pOutputs[index].nSize;
    }

    [[nodiscard]] uintmax_t get_shape_group_count() const {
        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        ::AX_U32 group = 0;
        if (const auto ret = ::AXCL_ENGINE_GetGroupIOInfoCount(this->handle_, &group); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model shape group count failed{0x%08X}.\n", ret);
            return 0;
        }
        return group;
    }

    [[nodiscard]] static std::string get_library_version() {
        int32_t major = 0, minor = 0, patch = 0;
        if (const auto ret = ::axclrtGetVersion(&major, &minor, &patch); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get library version failed{0x%08X}.\n", ret);
            return "";
        }
        return {std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch) + " " + ::AXCL_ENGINE_GetVersion()};
    }

    [[nodiscard]] std::string get_model_version() const {
        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return "";
        }
        return {::AXCL_ENGINE_GetModelToolsVersion(this->handle_)};
    }

    [[nodiscard]] int32_t get_model_type() const {
        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        ::AX_ENGINE_MODEL_TYPE_T type;
        if (const auto ret = ::AXCL_ENGINE_GetHandleModelType(this->handle_, &type); 0 != ret) {
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

        ::AX_ENGINE_NPU_ATTR_T attr{};
        if (const auto ret = ::AXCL_ENGINE_GetVNPUAttr(&attr); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get visual NPU type failed{0x%08X}.\n", ret);
            return -1;
        }
        return attr.eHardMode;
    }

    [[nodiscard]] int32_t get_batch_size() const {
        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        return static_cast<int32_t>(this->batch_);
    }

    [[nodiscard]] intmax_t get_sys_usage() const {
        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        return 0;
    }

    [[nodiscard]] intmax_t get_cmm_usage() const {
        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        ::AX_ENGINE_CMM_INFO usage{};
        if (const auto ret = ::AXCL_ENGINE_GetCMMUsage(this->handle_, &usage); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model usage failed{0x%08X}.\n", ret);
            return -1;
        }

        return usage.nCMMSize;
    }

private:
    uint32_t group_count_ = 0;
    uint32_t group_ = 0;
    uint32_t batch_ = 0;

    ::AX_ENGINE_HANDLE handle_{};
    ::AX_ENGINE_CONTEXT_T context_{};
    ::AX_ENGINE_IO_INFO_T* info_{};
    ::AX_ENGINE_IO_T io_{};
    std::vector<::AX_ENGINE_IO_BUFFER_T> inputs_;
    std::vector<::AX_ENGINE_IO_BUFFER_T> outputs_;

    bool is_initialized_ = false;
};
#endif
