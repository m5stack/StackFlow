/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include "middleware/ax_npu_runner_impl.hpp"

#if defined(ENV_AX_LOCAL_API_ENABLE)
middleware::npu_runner::npu_runner() {
    this->impl_ = new impl();
}

middleware::npu_runner::~npu_runner() {
    delete this->impl_;
}

bool middleware::npu_runner::init(const std::string& config_file, const uint32_t& index, const uint32_t& kind) {
    std::ignore = config_file;
    std::ignore = index;
    return this->impl_->init(kind);
}

bool middleware::npu_runner::final() {
    return this->impl_->final();
}

bool middleware::npu_runner::load(const std::string& model_path) {
    return this->impl_->load(model_path);
}

bool middleware::npu_runner::prepare(const bool& input_cached, const bool& output_cached, const uint32_t& group, const uint32_t& batch) {
    return this->impl_->prepare(input_cached, output_cached, group, batch);
}

bool middleware::npu_runner::run(const bool& parallel) {
    return this->impl_->run(parallel);
}

uint32_t middleware::npu_runner::get_input_count() const {
    return this->impl_->get_input_count();
}

uint32_t middleware::npu_runner::get_output_count() const {
    return this->impl_->get_output_count();
}

std::string middleware::npu_runner::get_input_name(const uint32_t& index) const {
    return this->impl_->get_input_name(index);
}

std::string middleware::npu_runner::get_output_name(const uint32_t& index) const {
    return this->impl_->get_output_name(index);
}

void *middleware::npu_runner::get_input_pointer(const uint32_t& index) const {
    return this->impl_->get_input_pointer(index);
}

void *middleware::npu_runner::get_output_pointer(const uint32_t& index) const {
    return this->impl_->get_output_pointer(index);
}

uintmax_t middleware::npu_runner::get_input_size(const uint32_t& index) const {
    return this->impl_->get_input_size(index);
}

uintmax_t middleware::npu_runner::get_output_size(const uint32_t& index) const {
    return this->impl_->get_output_size(index);
}

uintmax_t middleware::npu_runner::get_shape_group_count() const {
    return this->impl_->get_shape_group_count();
}

bool middleware::npu_runner::flush_input() const {
    return this->impl_->flush_input();
}

bool middleware::npu_runner::invalidate_output() const {
    return this->impl_->invalidate_output();
}

bool middleware::npu_runner::feed(const std::string& input_folder, const std::string& stimulus_name) const {
    return this->impl_->feed(input_folder, stimulus_name);
}

bool middleware::npu_runner::verify(const std::string& output_folder, const std::string& stimulus_name) const {
    return this->impl_->verify(output_folder, stimulus_name);
}

bool middleware::npu_runner::save(const std::string& output_folder, const std::string& stimulus_name) const {
    return this->impl_->save(output_folder, stimulus_name);
}

std::string middleware::npu_runner::get_library_version() const {
    return this->impl_->get_library_version();
}

std::string middleware::npu_runner::get_model_version() const {
    return this->impl_->get_model_version();
}

int32_t middleware::npu_runner::get_model_type() const {
    return this->impl_->get_model_type();
}

int32_t middleware::npu_runner::get_npu_type() const {
    return this->impl_->get_npu_type();
}

int32_t middleware::npu_runner::get_batch_size() const {
    return this->impl_->get_batch_size();
}

intmax_t middleware::npu_runner::get_sys_usage() const {
    return this->impl_->get_sys_usage();
}

intmax_t middleware::npu_runner::get_cmm_usage() const {
    return this->impl_->get_cmm_usage();
}

void middleware::npu_runner::sleep_for(const uint32_t sleep_duration) const {
    impl::sleep_for(sleep_duration);
}
#endif
