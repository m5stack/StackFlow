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
    class scalar_guard {
    public:
        scalar_guard() = delete;

        scalar_guard(T resource, std::function<void(T&)> destructor)
            : instance_(resource), destructor_(std::move(destructor)) {}

        scalar_guard(std::function<T()> resource_creator, std::function<void(T&)> destructor)
            : instance_(resource_creator()), destructor_(std::move(destructor)) {}

        scalar_guard(scalar_guard&& other) noexcept
            : instance_(std::exchange(other.instance_, T{})), destructor_(std::move(other.destructor_)) {}

        scalar_guard& operator=(scalar_guard&& other) noexcept {
            if (this != &other) {
                instance_ = std::exchange(other.instance_, T{});
                destructor_ = std::move(other.destructor_);
            }
            return *this;
        }

        scalar_guard(const scalar_guard&) = delete;
        scalar_guard& operator=(const scalar_guard&) = delete;

        ~scalar_guard() noexcept {
            if (destructor_) {
                destructor_(instance_);
            }
        }

        T& get() {
            return instance_;
        }

    private:
        T instance_;
        std::function<void(T&)> destructor_;
    };

}
