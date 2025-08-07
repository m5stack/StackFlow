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

#include <fstream>
#include <cctype>

#if defined(ENV_HAS_STD_FILESYSTEM)
#include <filesystem>
#endif // ENV_HAS_STD_FILESYSTEM
#if defined(ENV_HAS_POSIX_FILE_STAT)
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstring>
#include "utilities/scalar_guard.hpp"
#endif // ENV_HAS_POSIX_FILE_STAT
#if defined(ENV_HAS_WIN_API)
#include <Windows.h>
#endif // ENV_HAS_WIN_API

#include <string>

namespace utilities {
constexpr auto error_size = static_cast<uintmax_t>(-1);

enum class file_type {
    none,
    not_found,
    regular,
    directory,
    symlink,
    block,
    character,
    fifo,
    socket,
    unknown,
};

inline bool exists(const std::string& path) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::exists(path);
#elif defined(ENV_HAS_POSIX_FILE_STAT)
    struct ::stat path_stat{};
    return 0 == ::stat(path.c_str(), &path_stat);
#else
#error "Unsupported platform, native file system API is required."
#endif
}

inline uintmax_t file_size(const std::string& path) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::file_size(path);
#elif defined(ENV_HAS_POSIX_FILE_STAT)
    struct ::stat path_stat{};
    if (0 == ::stat(path.c_str(), &path_stat)) {
        return path_stat.st_size;
    }
    return error_size;
#else
#error "Unsupported platform, native file system API is required."
#endif
}

inline file_type status(const std::string& path) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    if (!exists(path)) {
        return file_type::not_found;
    }

    const auto status = std::filesystem::status(path);
    if (std::filesystem::is_regular_file(status)) {
        return file_type::regular;
    }
    if (std::filesystem::is_directory(status)) {
        return file_type::directory;
    }
    if (std::filesystem::is_symlink(status)) {
        return file_type::symlink;
    }
    if (std::filesystem::is_block_file(status)) {
        return file_type::block;
    }
    if (std::filesystem::is_character_file(status)) {
        return file_type::character;
    }
    if (std::filesystem::is_fifo(status)) {
        return file_type::fifo;
    }
    if (std::filesystem::is_socket(status)) {
        return file_type::socket;
    }
    return file_type::unknown;
#elif defined(ENV_HAS_POSIX_FILE_STAT)
    struct ::stat path_stat{};
    if (0 != ::stat(path.c_str(), &path_stat)) {
        return file_type::not_found;
    }

    if (S_ISREG(path_stat.st_mode)) {
        return file_type::regular;
    }
    if (S_ISDIR(path_stat.st_mode)) {
        return file_type::directory;
    }
    if (S_ISLNK(path_stat.st_mode)) {
        return file_type::symlink;
    }
    if (S_ISBLK(path_stat.st_mode)) {
        return file_type::block;
    }
    if (S_ISCHR(path_stat.st_mode)) {
        return file_type::character;
    }
    if (S_ISFIFO(path_stat.st_mode)) {
        return file_type::fifo;
    }
    if (S_ISSOCK(path_stat.st_mode)) {
        return file_type::socket;
    }
    return file_type::unknown;
#else
#error "Unsupported platform, native file system API is required."
#endif
}

inline bool is_regular_file(const std::string& path) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::is_regular_file(path);
#elif defined(ENV_HAS_POSIX_FILE_STAT)
    struct ::stat path_stat{};
    return 0 == ::stat(path.c_str(), &path_stat) && S_ISREG(path_stat.st_mode);
#else
#error "Unsupported platform, native file system API is required."
#endif
}

inline bool is_directory(const std::string& path) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::is_directory(path);
#elif defined(ENV_HAS_POSIX_FILE_STAT)
    struct ::stat path_stat{};
    return 0 == ::stat(path.c_str(), &path_stat) && S_ISDIR(path_stat.st_mode);
#else
#error "Unsupported platform, native file system API is required."
#endif
}

inline bool is_empty(const std::string& path) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::is_empty(path);
#elif defined(ENV_HAS_POSIX_FILE_STAT)
    if (!exists(path))
        return false;

    if (is_regular_file(path))
        return 0 == file_size(path);
    if (is_directory(path)) {
        auto dir = scalar_guard<DIR*>(::opendir(path.c_str()), ::closedir);
        if (nullptr == dir.get()) {
            return false;
        }

        struct ::dirent* entry;
        while ((entry = ::readdir(dir.get())) != nullptr) {
            if (0 != ::strcmp(entry->d_name, ".") && 0 != ::strcmp(entry->d_name, "..")) {
                return false;
            }
        }
        return true;
    }
    return false;
#else
#error "Unsupported platform, native file system API is required."
#endif
}

inline bool read(const std::string& file, void* data, const uintmax_t& size) {
    std::ifstream stream;
    stream.open(file, std::ios_base::binary | std::ios_base::in);
    if (stream.is_open()) {
        stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
        stream.close();
        return true;
    }
    return false;
}

inline bool write(const std::string& file, const void* data, const uintmax_t& size) {
    std::ofstream stream;
    stream.open(file, std::ios_base::binary | std::ios_base::out);
    if (stream.is_open()) {
        stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        stream.close();
        return true;
    }
    return false;
}

inline bool create_directory(const std::string& path) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::create_directories(path);
#elif defined(ENV_HAS_POSIX_FILE_STAT)
    return 0 == ::mkdir(path.c_str(), 0755);
#else
#error "Unsupported platform, native file system API is required."
#endif
}

inline std::string get_file_name(const std::string& file) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::path(file).filename().string();
#else
    if (const auto pos = file.find_last_of("/\\"); std::string::npos != pos) {
        return file.substr(pos + 1);
    }
    return file;
#endif
}

inline std::string get_file_extension(const std::string& file) {
#if defined(ENV_HAS_STD_FILESYSTEM)
    return std::filesystem::path(file).extension().string();
#else
    if (const auto pos = file.find_last_of('.'); std::string::npos != pos) {
        return file.substr(pos + 1);
    }
    return "";
#endif
}

inline std::string get_legal_name(const std::string& name) {
    std::string temp = name;

    size_t pos = temp.find_last_of("/\\");
    if (pos != std::string::npos) pos = 0;

    for (size_t i = pos; i < temp.size(); ++i)
        if (!(::isalnum(temp[i]) || '-' == temp[i] || '_' == temp[i])) temp[i] = '_';
    return temp;
}

}
