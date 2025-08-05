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
#include "middleware/ax_npu_runner.hpp"

#include "utilities/scalar_guard.hpp"
#include "utilities/file.hpp"
#include "utilities/file_mapper.hpp"
#include "utilities/log.hpp"

#include <ax_hrtimer.h>
#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include <mutex>

constexpr ::AX_U32 align_size = 128;
constexpr ::AX_S8 token_name[] = "toolkit";

struct middleware::npu_runner::impl {
    impl() = default;

    ~impl() {
        std::ignore = this->final();
    }

    [[nodiscard]] bool init(const uint32_t& kind) {
        std::lock_guard guard(this->mutex_);

        // 0. check the NPU kind
        if (kind >= ::AX_ENGINE_VIRTUAL_NPU_BUTT) {
            utilities::glog.print(utilities::log::type::error, "Specified NPU kind{%d} is out of range{total %d}.\n", kind, ::AX_ENGINE_VIRTUAL_NPU_BUTT);
            return false;
        }

        // 1. init ax system, using scalar_guard to ensure the finalization
        auto sys_guard = utilities::scalar_guard<::AX_S32>(
            ::AX_SYS_Init(),
            [](const ::AX_S32& state) {
                if (0 == state) {
                    std::ignore = ::AX_SYS_Deinit();
                }
            });

        // 2. check the initialization result
        if (0 != sys_guard.get()) {
            utilities::glog.print(utilities::log::type::error, "AX SYS init failed{0x%08X}.\n", sys_guard.get());
            return false;
        }
        utilities::glog.print(utilities::log::type::info, "AX SYS inited.\n");

        // 3. init ax engine
        ::AX_ENGINE_NPU_ATTR_T attr{};
        attr.eHardMode = static_cast<::AX_ENGINE_NPU_MODE_T>(kind);
        if (const int ret = ::AX_ENGINE_Init(&attr); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Init AX ENGINE as kind{%d} failed{0x%08X}.\n", kind, ret);
            return false;
        }
        utilities::glog.print(utilities::log::type::info, "AX ENGINE inited.\n");

        // 4. disable guard of env
        sys_guard.get() = -1;

        // 5. set the flag
        this->is_initialized_ = true;
        return true;
    }

    [[nodiscard]] bool final() {
        auto flag = true;

        std::lock_guard guard(this->mutex_);

        if (this->is_initialized_) {
            if (nullptr != this->handle_) {
                for (auto& input : inputs_) {
                    if (0 != input.phyAddr) {
                        flag &= 0 == ::AX_SYS_MemFree(input.phyAddr, input.pVirAddr);
                        input.phyAddr = 0;
                    }
                }
                for (auto& output : outputs_) {
                    if (0 != output.phyAddr) {
                        flag &= 0 == ::AX_SYS_MemFree(output.phyAddr, output.pVirAddr);
                        output.phyAddr = 0;
                    }
                }

                flag &= 0 == ::AX_ENGINE_DestroyHandle(this->handle_);
            }

            flag &= 0 == ::AX_ENGINE_Deinit();
            flag &= 0 == ::AX_SYS_Deinit();
            this->is_initialized_ = false;
        }

        return flag;
    }

    [[nodiscard]] bool load(const std::string& model_path) {
        std::lock_guard guard(this->mutex_);

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
            utilities::glog.print(utilities::log::type::error, "Model file{%s} error, please check it.\n", model_path.c_str());
            return false;
        }

        const auto model_map = utilities::file_mapper(model_path);
        if (nullptr == model_map.get()) {
            utilities::glog.print(utilities::log::type::error, "Model file{%s} mapping failed.\n", model_path.c_str());
            return false;
        }

        const auto model_name = utilities::get_file_name(model_path);
        ::AX_ENGINE_HANDLE_EXTRA_T extra_param{};
        extra_param.pName = reinterpret_cast<::AX_S8*>(const_cast<char*>(model_name.c_str()));
        if (const auto ret = ::AX_ENGINE_CreateHandleV2(&this->handle_, model_map.get(), model_map.size(), &extra_param); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Create model{%s} handle failed.\n", model_path.c_str());
            return false;
        }

        if (const auto ret = ::AX_ENGINE_CreateContextV2(this->handle_, &this->context_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Create model{%s} context failed.\n", model_path.c_str());
            return false;
        }

        return true;
    }

    [[nodiscard]] bool prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) {
        std::lock_guard guard(this->mutex_);

        // 0. check the handle
        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is null.\n");
            return false;
        }

        // 1. get the count of shape group
        if (const auto ret = ::AX_ENGINE_GetGroupIOInfoCount(this->handle_, &this->group_); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model shape group count failed{0x%08X}.\n", ret);
            return false;
        }

        // 2. check the group index
        if (group >= this->group_) {
            utilities::glog.print(utilities::log::type::error, "Model group{%d} is out of range{total %d}.\n", group, this->group_);
            return false;
        }

        // 3. get the IO info
        if (1 == this->group_) {
            if (const auto ret = ::AX_ENGINE_GetIOInfo(this->handle_, &info_); 0 != ret) {
                utilities::glog.print(utilities::log::type::error, "Get model IO info failed{0x%08X}.\n", ret);
                return false;
            }
        } else {
            if (const auto ret = ::AX_ENGINE_GetGroupIOInfo(this->handle_, group, &info_); 0 != ret) {
                utilities::glog.print(utilities::log::type::error, "Get model group{index: %d} IO info failed{0x%08X}.\n", group, ret);
                return false;
            }
        }

        // 4. check the batch size
        this->batch = (0 == batch ? 1 : batch);
        this->io_.nBatchSize = this->batch;

        // 5. prepare the input and output
        this->inputs_.resize(this->info_->nInputSize, ::AX_ENGINE_IO_BUFFER_T{});
        this->io_.pInputs = this->inputs_.data();
        this->io_.nInputSize = this->info_->nInputSize;

        this->outputs_.resize(this->info_->nOutputSize, ::AX_ENGINE_IO_BUFFER_T{});
        this->io_.pOutputs = this->outputs_.data();
        this->io_.nOutputSize = this->info_->nOutputSize;

        this->input_cached_.resize(inputs_.size(), input_cached);
        this->output_cached_.resize(outputs_.size(), output_cached);

        // 6. prepare the memory, inputs
        for (::AX_U32 i = 0; i < this->info_->nInputSize; i++) {
            const auto& meta = this->info_->pInputs[i];
            auto& io = this->io_.pInputs[i];
            io.nSize = meta.nSize * this->batch;

            if (this->input_cached_[i]) {
                if (const auto ret = ::AX_SYS_MemAllocCached(&io.phyAddr, &io.pVirAddr, io.nSize, align_size, token_name); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Memory allocation for tensor {%s} failed{0x%08X}.\n", meta.pName, ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "Memory for tensor {%s} is allocated.\n", meta.pName);
            } else {
                if (const auto ret = ::AX_SYS_MemAlloc(&io.phyAddr, &io.pVirAddr, io.nSize, align_size, token_name); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Memory allocation for tensor {%s} failed{0x%08X}.\n", meta.pName, ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "Memory for tensor {%s} is allocated.\n", meta.pName);
            }

            // clean memory, some cases model may need to clean memory
            memset(io.pVirAddr, 0, io.nSize);

            if (this->output_cached_[i]) {
                if (const auto ret = ::AX_SYS_MflushCache(io.phyAddr, io.pVirAddr, io.nSize); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Memory flush cache for tensor {%s} failed{0x%08X}.\n", meta.pName, ret);
                    return false;
                }
            }

            utilities::glog.print(utilities::log::type::info, "Set tensor { name: %s, phy: %p, vir: %p, size: %u Bytes }.\n",
                meta.pName, reinterpret_cast<void*>(io.phyAddr), io.pVirAddr, io.nSize);
        }

        // 7. prepare the memory, outputs
        for (::AX_U32 i = 0; i < this->info_->nOutputSize; i++) {
            const auto& meta = this->info_->pOutputs[i];
            auto& io = this->io_.pOutputs[i];
            io.nSize = meta.nSize * this->batch;

            if (this->output_cached_[i]) {
                if (const auto ret = ::AX_SYS_MemAllocCached(&io.phyAddr, &io.pVirAddr, io.nSize, align_size, token_name); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Memory allocation for tensor {%s} failed{0x%08X}.\n", meta.pName, ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "Memory for tensor {%s} is allocated.\n", meta.pName);
            } else {
                if (const auto ret = ::AX_SYS_MemAlloc(&io.phyAddr, &io.pVirAddr, io.nSize, align_size, token_name); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Memory allocation for tensor {%s} failed{0x%08X}.\n", meta.pName, ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "Memory for tensor {%s} is allocated.\n", meta.pName);
            }

            // clean memory, some cases model may need to clean memory
            memset(io.pVirAddr, 0, io.nSize);

            if (this->output_cached_[i]) {
                if (const auto ret = ::AX_SYS_MflushCache(io.phyAddr, io.pVirAddr, io.nSize); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Memory flush cache for tensor {%s} failed{0x%08X}.\n", meta.pName, ret);
                    return false;
                }
            }

            utilities::glog.print(utilities::log::type::info, "Set tensor { name: %s, phy: %p, vir: %p, size: %u Bytes }.\n",
                meta.pName, reinterpret_cast<void*>(io.phyAddr), io.pVirAddr, io.nSize);
        }

        return true;
    }

    [[nodiscard]] bool run(const bool& parallel) {
        std::lock_guard guard(this->mutex_);

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
        if (1 == this->group_) {
            ret = ::AX_ENGINE_RunSyncV2(this->handle_, this->context_, &this->io_);
        } else {
            ret = ::AX_ENGINE_RunGroupIOSync(this->handle_, this->context_, this->group_, &this->io_);
        }
        if (0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Run model failed{0x%08X}.\n", ret);
            return false;
        }
        utilities::glog.print(utilities::log::type::info, "Running done.\n");

        return true;
    }

    [[nodiscard]] uint32_t get_input_count() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        return this->info_->nInputSize;
    }

    [[nodiscard]] uint32_t get_output_count() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        return this->info_->nOutputSize;
    }

    [[nodiscard]] std::string get_input_name(const uint32_t& index) {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return {};
        }

        if (index >= this->info_->nInputSize) {
            return {};
        }
        return std::string{this->info_->pInputs[index].pName};
    }

    [[nodiscard]] std::string get_output_name(const uint32_t& index) {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return {};
        }

        if (index >= this->info_->nOutputSize) {
            return {};
        }
        return std::string{this->info_->pOutputs[index].pName};
    }

    [[nodiscard]] void *get_input_pointer(const uint32_t& index) {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return nullptr;
        }

        if (index >= this->info_->nInputSize) {
            return nullptr;
        }
        return reinterpret_cast<void *>(this->io_.pInputs[index].phyAddr);
    }

    [[nodiscard]] void *get_output_pointer(const uint32_t& index) {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return nullptr;
        }

        if (index >= this->info_->nOutputSize) {
            return nullptr;
        }
        return reinterpret_cast<void *>(this->io_.pOutputs[index].phyAddr);
    }

    [[nodiscard]] uintmax_t get_input_size(const uint32_t& index) {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        if (index >= this->info_->nInputSize) {
            return 0;
        }
        return this->io_.pInputs[index].nSize;
    }

    [[nodiscard]] uintmax_t get_output_size(const uint32_t& index) {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        if (index >= this->info_->nOutputSize) {
            return 0;
        }
        return this->io_.pOutputs[index].nSize;
    }

    [[nodiscard]] uintmax_t get_shape_group_count() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->info_) {
            utilities::glog.print(utilities::log::type::error, "Model io info is not set, prepare first.\n");
            return 0;
        }

        return this->group_;
    }

    [[nodiscard]] bool flush_input() {
        std::lock_guard guard(this->mutex_);

        for (::AX_U32 i = 0; i < this->io_.nInputSize; i++) {
            const auto buffer = &(this->io_.pInputs[i]);
            if (const auto ret = ::AX_SYS_MflushCache(buffer->phyAddr, buffer->pVirAddr, buffer->nSize); 0 != ret) {
                utilities::glog.print(utilities::log::type::warn, "Flush input{index: %d} failed{0x%08X}.\n", i, ret);
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool invalidate_output() {
        std::lock_guard guard(this->mutex_);

        for (::AX_U32 i = 0; i < this->io_.nOutputSize; i++) {
            const auto buffer = &(this->io_.pOutputs[i]);
            if (const auto ret = ::AX_SYS_MinvalidateCache(buffer->phyAddr, buffer->pVirAddr, buffer->nSize); 0 != ret) {
                utilities::glog.print(utilities::log::type::warn, "Invalidate output{index: %d} failed{0x%08X}.\n", i, ret);
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool feed(const std::string& input_folder, const std::string& stimulus_name) {
        for (uint32_t i = 0; i < this->get_input_count(); i++) {
            const auto file_name = utilities::get_legal_name(this->get_input_name(i));
            auto file_path = input_folder;
            file_path.append("/").append(stimulus_name).append("/").append(file_name).append(".bin");

            if (!read(file_path, this->inputs_[i].pVirAddr, this->inputs_[i].nSize)) {
                utilities::glog.print(utilities::log::type::error, "Read tensor {idx: %d, name: %s} file {%s} failed.\n", i, file_name.c_str(), file_path.c_str());
                return false;
            }

            if (this->input_cached_[i]) {
                if (const auto ret = ::AX_SYS_MflushCache(this->inputs_[i].phyAddr, this->inputs_[i].pVirAddr, this->inputs_[i].nSize); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Flush tensor {idx: %d, name: %s} cache failed{%d}.\n", i, file_name.c_str(), ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "Tensor {idx: %d, name: %s} cache flushed.\n", i, file_name.c_str());
            }
        }
        return true;
    }

    [[nodiscard]] bool verify(const std::string& output_folder, const std::string& stimulus_name) {
        for (uint32_t i = 0; i < this->get_output_count(); i++) {
            const auto file_name = utilities::get_legal_name(this->get_output_name(i));
            auto file_path = output_folder;
            file_path.append("/").append(stimulus_name).append("/").append(file_name).append(".bin");

            std::vector<uint8_t> file_buffer(this->get_output_size(i));
            if (!read(file_path, file_buffer.data(), file_buffer.size())) {
                utilities::glog.print(utilities::log::type::error, "Read tensor {idx: %d, name: %s} file {%s} failed.\n", i, file_name.c_str(), file_path.c_str());
                return false;
            }

            if (this->output_cached_[i]) {
                if (const auto ret = ::AX_SYS_MinvalidateCache(this->outputs_[i].phyAddr, this->outputs_[i].pVirAddr, this->outputs_[i].nSize); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Invalidate tensor {idx: %d, name: %s} cache failed{%d}.\n", i, file_name.c_str(), ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "Tensor {idx: %d, name: %s} cache invalidated.\n", i, file_name.c_str());
            }

            if (!runner::verify(file_buffer.data(), this->outputs_[i].pVirAddr, this->outputs_[i].nSize)) {
                utilities::glog.print(utilities::log::type::error, "Verify tensor {idx: %d, name: %s} failed.\n", i, file_name.c_str());
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool save(const std::string& output_folder, const std::string& stimulus_name) {
        auto file_folder = output_folder;
        file_folder.append("/").append(stimulus_name);

        if (!utilities::exists(file_folder) && !utilities::create_directory(file_folder)) {
            utilities::glog.print(utilities::log::type::error, "Create folder {%s} failed.\n", file_folder.c_str());
            return false;
        }

        for (uint32_t i = 0; i < this->get_output_count(); i++) {
            const auto file_name = utilities::get_legal_name(this->get_output_name(i));
            auto file_path = file_folder;
            file_path.append("/").append(file_name).append(".bin");

            if (this->output_cached_[i]) {
                if (const auto ret = ::AX_SYS_MinvalidateCache(this->outputs_[i].phyAddr, this->outputs_[i].pVirAddr, this->outputs_[i].nSize); 0 != ret) {
                    utilities::glog.print(utilities::log::type::error, "Invalidate tensor {idx: %d, name: %s} cache failed{%d}.\n", i, file_name.c_str(), ret);
                    return false;
                }
                utilities::glog.print(utilities::log::type::info, "Tensor {idx: %d, name: %s} cache invalidated.\n", i, file_name.c_str());
            }

            if (!write(file_path, this->outputs_[i].pVirAddr, this->outputs_[i].nSize)) {
                utilities::glog.print(utilities::log::type::error, "Write tensor {idx: %d, name: %s} file {%s} failed.\n", i, file_name.c_str(), file_path.c_str());
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::string get_library_version() {
        std::lock_guard guard(this->mutex_);

        const auto ver = ::AX_ENGINE_GetVersion();
        if (nullptr == ver) {
            utilities::glog.print(utilities::log::type::error, "Get library version failed.\n");
            return "";
        }
        return {ver};
    }

    [[nodiscard]] std::string get_model_version() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model id is not set, load model first.\n");
            return "";
        }
        return {::AX_ENGINE_GetModelToolsVersion(this->handle_)};
    }

    [[nodiscard]] int32_t get_model_type() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        ::AX_ENGINE_MODEL_TYPE_T type;
        if (const auto ret = ::AX_ENGINE_GetHandleModelType(this->handle_, &type); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model type failed{0x%08X}.\n", ret);
            return -1;
        }
        return type;
    }

    [[nodiscard]] int32_t get_npu_type() {
        std::lock_guard guard(this->mutex_);

        if (!this->is_initialized_) {
            utilities::glog.print(utilities::log::type::error, "axcl is not initialized.\n");
            return false;
        }

        ::AX_ENGINE_NPU_ATTR_T attr{};
        if (const auto ret = ::AX_ENGINE_GetVNPUAttr(&attr); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get visual NPU type failed{0x%08X}.\n", ret);
            return -1;
        }
        return attr.eHardMode;
    }

    [[nodiscard]] int32_t get_batch_size() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        return static_cast<int32_t>(this->batch);
    }

    [[nodiscard]] intmax_t get_sys_usage() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        return 0;
    }

    [[nodiscard]] intmax_t get_cmm_usage() {
        std::lock_guard guard(this->mutex_);

        if (nullptr == this->handle_) {
            utilities::glog.print(utilities::log::type::error, "Model handle is not set, load model first.\n");
            return -1;
        }

        ::AX_ENGINE_CMM_INFO usage{};
        if (const auto ret = ::AX_ENGINE_GetCMMUsage(this->handle_, &usage); 0 != ret) {
            utilities::glog.print(utilities::log::type::error, "Get model usage failed{0x%08X}.\n", ret);
            return -1;
        }

        return usage.nCMMSize;
    }

    static void sleep_for(const uint32_t sleep_duration) {
        ::AX_SYS_Msleep(sleep_duration);
    }

private:
    uint32_t group_ = 0;
    uint32_t batch = 0;

    ::AX_ENGINE_HANDLE handle_{};
    ::AX_ENGINE_CONTEXT_T context_{};
    ::AX_ENGINE_IO_INFO_T* info_{};
    ::AX_ENGINE_IO_T io_{};
    std::vector<::AX_ENGINE_IO_BUFFER_T> inputs_;
    std::vector<::AX_ENGINE_IO_BUFFER_T> outputs_;

    std::vector<bool> input_cached_;
    std::vector<bool> output_cached_;

    bool is_initialized_ = false;
    std::mutex mutex_;
};
#endif
