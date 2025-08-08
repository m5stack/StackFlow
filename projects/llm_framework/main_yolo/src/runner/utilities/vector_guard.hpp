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

#include <functional>
#include <vector>
#include <utility>

namespace utilities {

    template<typename T>
    class vector_guard {
    public:
        vector_guard() = delete;

        vector_guard(std::vector<T> resources, std::function<void(T&)> destructor)
            : instance_(std::move(resources)), destructor_(std::move(destructor)) {}

        vector_guard(std::function<std::vector<T>()> resource_creator, std::function<void(T&)> destructor)
            : instance_(resource_creator()), destructor_(std::move(destructor)) {}

        vector_guard(vector_guard&& other) noexcept
            : instance_(std::exchange(other.instance_, std::vector<T>{})), destructor_(std::move(other.destructor_)) {}

        vector_guard& operator=(vector_guard&& other) noexcept {
            if (this != &other) {
                instance_ = std::exchange(other.instance_, std::vector<T>{});
                destructor_ = std::move(other.destructor_);
            }
            return *this;
        }

        vector_guard(const vector_guard&) = delete;
        vector_guard& operator=(const vector_guard&) = delete;

        ~vector_guard() noexcept {
            if (destructor_) {
                for (auto& resource : instance_) {
                    destructor_(resource);
                }
            }
        }

        std::vector<T>& get() {
            return instance_;
        }

        T* data() {
            return instance_.data();
        }

    private:
        std::vector<T> instance_;
        std::function<void(T&)> destructor_;
    };

}
