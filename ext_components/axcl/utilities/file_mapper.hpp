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

#if defined(ENV_HAS_POSIX_FILE_STAT)
#include "utilities/scalar_guard.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif // ENV_HAS_POSIX_FILE_STAT

#if defined(ENV_HAS_WIN_API)
#include <Windows.h>
#endif // ENV_HAS_WIN_API

#include <utilities/file.hpp>

namespace utilities {

struct file_mapper {
    file_mapper() = delete;

    explicit file_mapper(const std::string& path) {
#if defined(ENV_HAS_POSIX_FILE_STAT)

        auto fd_guard = scalar_guard<int>(
            ::open(path.c_str(), O_RDONLY),
            [](const int& fd) { if (fd != -1) ::close(fd); }
        );

        if (-1 == fd_guard.get()) {
            return;
        }

        auto size = file_size(path);
        auto map_guard = scalar_guard<void*>(
            ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd_guard.get(), 0),
            [&size](void*& addr) { if (MAP_FAILED != addr && nullptr != addr) ::munmap(addr, size); }
        );

        if (MAP_FAILED == map_guard.get()) {
            return;
        }

        this->size_ = size;
        std::swap(this->fd_, fd_guard.get());
        std::swap(this->buffer_, map_guard.get());
#endif // ENV_HAS_POSIX_FILE_STAT
    }

    ~file_mapper() {
#if defined(ENV_HAS_POSIX_FILE_STAT)
        if (nullptr != this->buffer_) {
            ::munmap(this->buffer_, this->size_);
            this->buffer_ = nullptr;
        }
        if (-1 != this->fd_) {
            ::close(this->fd_);
            this->fd_ = -1;
        }
#endif // ENV_HAS_POSIX_FILE_STAT
    }

    [[nodiscard]] void* get() const {
        return this->buffer_;
    }

    [[nodiscard]] uint64_t size() const {
        return this->size_;
    }

private:
    int fd_ = -1;
    void* buffer_ = nullptr;
    uint64_t size_ = 0;
};

}
