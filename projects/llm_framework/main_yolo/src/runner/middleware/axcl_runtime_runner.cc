/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include "middleware/axcl_runtime_runner_impl.hpp"

#if defined(CONFIG_AXCL_RUNTIME_API_ENABLE)
middleware::runtime_runner::runtime_runner() {
    this->impl_ = new impl();
}

middleware::runtime_runner::~runtime_runner() {
    delete this->impl_;
}

bool middleware::runtime_runner::init(const std::string& config_file, const uint32_t& device_index, const uint32_t& kind) {
    return this->impl_->init(config_file, device_index, kind);
}

bool middleware::runtime_runner::final() {
    return this->impl_->final();
}

bool middleware::runtime_runner::load(const std::string& model_path) {
    return this->impl_->load(model_path);
}

bool middleware::runtime_runner::prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) {
    return this->impl_->prepare(input_cached, output_cached, group, batch);
}

bool middleware::runtime_runner::run(const bool& parallel) {
    return this->impl_->run(parallel);
}

bool middleware::runtime_runner::set() {
    return this->impl_->set();
}

uint32_t middleware::runtime_runner::get_input_count() const {
    return this->impl_->get_input_count();
}

uint32_t middleware::runtime_runner::get_output_count() const {
    return this->impl_->get_output_count();
}

std::string middleware::runtime_runner::get_input_name(const uint32_t& index) const {
    return this->impl_->get_input_name(index);
}

std::string middleware::runtime_runner::get_output_name(const uint32_t& index) const {
    return this->impl_->get_output_name(index);
}

void *middleware::runtime_runner::get_input_pointer(const uint32_t& index) const {
    return this->impl_->get_input_pointer(index);
}

void *middleware::runtime_runner::get_output_pointer(const uint32_t& index) const {
    return this->impl_->get_output_pointer(index);
}

uintmax_t middleware::runtime_runner::get_input_size(const uint32_t& index) const {
    return this->impl_->get_input_size(index);
}

uintmax_t middleware::runtime_runner::get_output_size(const uint32_t& index) const {
    return this->impl_->get_output_size(index);
}

uintmax_t middleware::runtime_runner::get_shape_group_count() const {
    return this->impl_->get_shape_group_count();
}

std::string middleware::runtime_runner::get_library_version() const {
    return impl::get_library_version();
}

std::string middleware::runtime_runner::get_model_version() const {
    return this->impl_->get_model_version();
}

int32_t middleware::runtime_runner::get_model_type() const {
    return this->impl_->get_model_type();
}

int32_t middleware::runtime_runner::get_npu_type() const {
    return this->impl_->get_npu_type();
}

int32_t middleware::runtime_runner::get_batch_size() const {
    return this->impl_->get_batch_size();
}

intmax_t middleware::runtime_runner::get_sys_usage() const {
    return this->impl_->get_sys_usage();
}

intmax_t middleware::runtime_runner::get_cmm_usage() const {
    return this->impl_->get_cmm_usage();
}
#endif
