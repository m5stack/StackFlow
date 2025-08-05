/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include "middleware/axcl_base.hpp"


#if defined(ENV_AXCL_RUNTIME_API_ENABLE) || defined(ENV_AXCL_NATIVE_API_ENABLE)

#include "utilities/scalar_guard.hpp"
#include "utilities/file.hpp"
#include "utilities/log.hpp"

#include <axcl.h>

#include <mutex>

static int static_module_count{0};
static std::mutex static_module_mutex;

static bool module_init(const std::string& config, const uint32_t& index, const middleware::axcl_base::npu_func& func) {
    // 1. init axcl, using scalar_guard to ensure the finalization
    const auto cfg_flag = utilities::exists(config) && utilities::is_regular_file(config);
    utilities::glog.print(utilities::log::type::info, "axcl initializing...\n");
    auto env_guard = utilities::scalar_guard<int32_t>(
        axclInit(cfg_flag ? config.c_str() : nullptr),
        [](const int32_t& code) {
            if (0 == code) {
                std::ignore = axclFinalize();
            }
        }
    );

    // 2. check the initialization result
    if (const int ret = env_guard.get(); 0 != ret) {
        utilities::glog.print(utilities::log::type::error, "Init axcl failed{0x%08X}.\n", ret);
        return false;
    }
    utilities::glog.print(utilities::log::type::info, "axcl inited.\n");

    // 3. get device list
    axclrtDeviceList lst;
    if (const auto ret = axclrtGetDeviceList(&lst); 0 != ret || 0 == lst.num) {
        utilities::glog.print(utilities::log::type::error,
            "Get axcl device failed{0x%08X}, find total %d device.\n", ret, lst.num);
        return false;
    }

    // 4. check the device index
    if (index >= lst.num) {
        utilities::glog.print(utilities::log::type::error,
            "Specified device index{%d} is out of range{total %d}.\n", index, lst.num);
        return false;
    }

    // 5. set device
    if (const auto ret = axclrtSetDevice(lst.devices[index]); 0 != ret) {
        utilities::glog.print(utilities::log::type::error,
            "Set axcl device as index{%d} failed{0x%08X}.\n", index, ret);
        return false;
    }
    utilities::glog.print(utilities::log::type::info,
        "Select axcl device{index: %d} as {%d}.\n", index, lst.devices[index]);

    // 6. init NPU
    if (!func()) {
        utilities::glog.print(utilities::log::type::error, "Init NPU failed.\n");
        return false;
    }

    // 7. disengage guard of env
    env_guard.get() = -1;
    return true;
}

static bool module_final(const middleware::axcl_base::npu_func& func) {
    const auto flag = func();
    const auto ret = axclFinalize();
    return flag && (0 == ret);
}

bool middleware::axcl_base::init(const std::string &config, const uint32_t &index, const npu_func &func) {
    bool flag = false;
    {
        std::lock_guard lock(static_module_mutex);
        if (static_module_count == 0) {
            flag = module_init(config, index, func);
            if (flag) {
                static_module_count++;
            }
        } else {
            static_module_count++;
            flag = true;
        }
    }

    return flag;
}

bool middleware::axcl_base::final(const npu_func& func) {
    bool flag = false;
    {
        std::lock_guard lock(static_module_mutex);
        if (static_module_count == 1) {
            flag = module_final(func);
            if (flag) {
                static_module_count--;
            }
        } else if (static_module_count > 1) {
            static_module_count--;
            flag = true;
        }
    }

    return flag;
}

bool middleware::axcl_base::flush_input() const {
    return true;
}

bool middleware::axcl_base::invalidate_output() const {
    return true;
}

[[nodiscard]] bool middleware::axcl_base::feed(const std::string& input_folder, const std::string& stimulus_name) const {
    for (uint32_t i = 0; i < this->get_input_count(); i++) {
        if (const auto ret = feed(input_folder, stimulus_name, this->get_input_name(i), this->get_input_pointer(i), this->get_input_size(i) * this->get_batch_size()); !ret) {
            utilities::glog.print(utilities::log::type::error, "Feed tensor {idx: %d, name: %s} failed.\n", i, this->get_input_name(i).c_str());
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool middleware::axcl_base::verify(const std::string& output_folder, const std::string& stimulus_name) const {
    for (uint32_t i = 0; i < this->get_output_count(); i++) {
        if (const auto ret = verify(output_folder, stimulus_name, this->get_output_name(i),this->get_output_pointer(i), this->get_output_size(i) * this->get_batch_size()); !ret) {
            utilities::glog.print(utilities::log::type::error, "Verify tensor {idx: %d, name: %s} failed.\n", i, this->get_output_name(i).c_str());
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool middleware::axcl_base::save(const std::string& output_folder, const std::string& stimulus_name) const {
    auto file_folder = output_folder;
    file_folder.append("/").append(stimulus_name);

    if (!utilities::exists(file_folder) && !utilities::create_directory(file_folder)) {
        utilities::glog.print(utilities::log::type::error, "Create folder {%s} failed.\n", file_folder.c_str());
        return false;
    }

    for (uint32_t i = 0; i < this->get_output_count(); i++) {
        if (const auto ret = save(output_folder, stimulus_name, this->get_output_name(i), this->get_output_pointer(i), this->get_output_size(i) * this->get_batch_size()); !ret) {
            utilities::glog.print(utilities::log::type::error, "Save tensor {idx: %d, name: %s} failed.\n", i, this->get_output_name(i).c_str());
            return false;
        }
    }
    return true;
}

bool middleware::axcl_base::feed(const std::string& folder, const std::string& stimulus_name, const std::string& tensor_name, void* address, const uintmax_t size) {
    const auto file_name = utilities::get_legal_name(tensor_name);
    auto file_path(folder);
    file_path.append("/").append(stimulus_name).append("/").append(file_name).append(".bin");

    std::vector<uint8_t> tensor_buffer(size);
    if (!read(file_path, tensor_buffer.data(), tensor_buffer.size())) {
        utilities::glog.print(utilities::log::type::error,
            "Read tensor {name: %s} stimulus file {%s} failed.\n", tensor_name.c_str(), file_path.c_str());
        return false;
    }

    if (const auto ret = axclrtMemcpy(address, tensor_buffer.data(), size, AXCL_MEMCPY_HOST_TO_DEVICE); 0 != ret) {
        utilities::glog.print(utilities::log::type::error,
            "Receive tensor {name: %s, addr: 0x%08X, size: %ld} from host {addr: 0x%08X} failed{0x%08X}.\n",
            tensor_name.c_str(), address, size, tensor_buffer.data(), ret);
        return false;
    }

    return true;
}

bool middleware::axcl_base::verify(const std::string& folder, const std::string& stimulus_name, const std::string& tensor_name, void* address, const uintmax_t size) {
    const auto file_name = utilities::get_legal_name(tensor_name);
    auto file_path(folder);
    file_path.append("/").append(stimulus_name).append("/").append(file_name).append(".bin");

    std::vector<uint8_t> file_buffer(size);
    if (!read(file_path, file_buffer.data(), file_buffer.size())) {
        utilities::glog.print(utilities::log::type::error,
            "Read tensor {name: %s} file {%s} failed.\n", file_name.c_str(), file_path.c_str());
        return false;
    }

    std::vector<uint8_t> tensor_buffer(size);

    if (const auto ret = axclrtMemcpy(tensor_buffer.data(), address, size, AXCL_MEMCPY_DEVICE_TO_HOST); 0 != ret) {
        utilities::glog.print(utilities::log::type::error,
            "Send tensor {name: %s, addr: 0x%08X, size: %ld} to host {addr: 0x%08X} failed{0x%08X}.\n",
            file_name.c_str(), address, size, tensor_buffer.data(), ret);
        return false;
    }

    if (!runner::verify(file_buffer.data(), tensor_buffer.data(), file_buffer.size())) {
        utilities::glog.print(utilities::log::type::error, "Verify tensor {name: %s} failed.\n", file_name.c_str());
        return false;
    }

    return true;
}

bool middleware::axcl_base::save(const std::string& output_folder, const std::string& stimulus_name, const std::string& tensor_name, void* address, const uintmax_t size) {
    const auto file_name = utilities::get_legal_name(tensor_name);
    auto file_path(output_folder);
    file_path.append("/").append(stimulus_name).append("/").append(file_name).append(".bin");

    std::vector<uint8_t> tensor_buffer(size);
    if (const auto ret = axclrtMemcpy(tensor_buffer.data(), address, size, AXCL_MEMCPY_DEVICE_TO_HOST); 0 != ret) {
        utilities::glog.print(utilities::log::type::error,
            "Send tensor {name: %s, addr: 0x%08X, size: %ld} to host {addr: 0x%08X} failed{0x%08X}.\n",
            file_name.c_str(), address, size, tensor_buffer.data(), ret);
        return false;
    }

    if (!write(file_path, tensor_buffer.data(), tensor_buffer.size())) {
        utilities::glog.print(utilities::log::type::error, "Write tensor {name: %s} file {%s} failed.\n", file_name.c_str(), file_path.c_str());
        return false;
    }

    return true;
}
#endif
