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

#include <chrono>

namespace utilities {

class timer {
public:
    using nanoseconds = std::chrono::nanoseconds;
    using microseconds = std::chrono::microseconds;
    using milliseconds = std::chrono::milliseconds;
    using seconds = std::chrono::seconds;
    using minutes = std::chrono::minutes;
    using hours = std::chrono::hours;

    timer() {
        start();
    }

    void start() {
        stop();
        this->start_ = this->end_;
    }

    void stop() {
#ifdef _MSC_VER
        this->end_ = std::chrono::system_clock::now();
#else
        this->end_ = std::chrono::high_resolution_clock::now();
#endif
    }

    template<typename T>
    float elapsed() {
        if (this->end_ <= this->start_) {
            this->stop();
        }

        if (std::is_same_v<T, std::chrono::nanoseconds>) {
            const auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(this->end_ - this->start_).count();
            return static_cast<float>(t);
        }
        if (std::is_same_v<T, std::chrono::microseconds>) {
            const auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(this->end_ - this->start_).count();
            return static_cast<float>(t) / 1000.f;
        }
        if (std::is_same_v<T, std::chrono::milliseconds>) {
            const auto t = std::chrono::duration_cast<std::chrono::microseconds>(this->end_ - this->start_).count();
            return static_cast<float>(t) / 1000.f;
        }
        if (std::is_same_v<T, std::chrono::seconds>) {
            const auto t = std::chrono::duration_cast<std::chrono::milliseconds>(this->end_ - this->start_).count();
            return static_cast<float>(t) / 1000.f;
        }
        if (std::is_same_v<T, std::chrono::minutes>) {
            const auto t = std::chrono::duration_cast<std::chrono::milliseconds>(this->end_ - this->start_).count();
            return static_cast<float>(t) / (60.f * 1000.f);
        }
        if (std::is_same_v<T, std::chrono::hours>) {
            const auto t = std::chrono::duration_cast<std::chrono::seconds>(this->end_ - this->start_).count();
            return static_cast<float>(t) / (60.f * 60.f);
        }
        return 0.f;
    }

    float elapsed() {
        return elapsed<std::chrono::milliseconds>();
    }

private:
    std::chrono::system_clock::time_point start_, end_;
};

}
