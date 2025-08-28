/*
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 * Author: wanglusheng@axera-tech.com
 */

#pragma once

#include <cstdarg>
#include <cstdio>

namespace utilities {
    class log
    {
    public:
        enum class type {
            debug = 0,
            info  = 1,
            warn = 2,
            error = 3,
        };
        
        log() = default;
        explicit log(const type& mode) : type_(mode) {}
        ~log() = default;

        void print(const type& mode, const char *fmt, ...) const {
            if (mode >= this->type_) {
                switch (mode) {
                    case type::debug:
                        ::fprintf(stdout, "[DEBUG] ");
                    break;
                    case type::info:
                        ::fprintf(stdout, "[INFO] ");
                    break;
                    case type::warn:
                        ::fprintf(stdout, "[WARNING] ");
                    break;
                    case type::error:
                        ::fprintf(stdout, "[ERROR] ");
                    break;
                }

                ::va_list args;
                va_start(args, fmt);
                ::vprintf( fmt, args);
                va_end(args);
            }
            ::fflush(stdout);
        }
        void set_level(const type& mode) {
            this->type_ = mode;
        }

    private:
        type type_ = type::warn;
    };

    static log glog;
}
