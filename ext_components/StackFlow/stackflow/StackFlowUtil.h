/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <string>
#include <cstring>
#include <unordered_map>
#include <list>
#include <vector>
#include <functional>
#include "pzmq.hpp"
#include <memory>
#define WORK_ID_NONE -100

#define RPC_PUSH_PARAM(_obj, _data1, _data2)                                                     \
    do {                                                                                         \
        _obj.resize(1 + _data1.size() + _data2.size());                                          \
        _obj[0] = (unsigned char)_data1.size();                                                  \
        memcpy((void *)(_obj.data() + 1), (void *)_data1.data(), _data1.size());                 \
        memcpy((void *)(_obj.data() + _data1.size() + 1), (void *)_data2.data(), _data2.size()); \
    } while (0)

#define RPC_PARSE_TO_FIRST(_obj)  _obj.substr(1, static_cast<size_t>(_obj[0]))
#define RPC_PARSE_TO_SECOND(_obj) _obj.substr(static_cast<size_t>(_obj[0]) + 1)

namespace StackFlows {
std::string sample_json_str_get(const std::string &json_str, const std::string &json_key);
int sample_get_work_id_num(const std::string &work_id);
std::string sample_get_work_id_name(const std::string &work_id);
std::string sample_get_work_id(int work_id_num, const std::string &unit_name);
std::string sample_escapeString(const std::string &input);
std::string sample_unescapeString(const std::string &input, bool ucs2 = false);
bool decode_stream(const std::string &in, std::string &out, std::unordered_map<int, std::string> &stream_buff);
int decode_base64(const std::string &in, std::string &out);
int encode_base64(const std::string &in, std::string &out);
std::string unit_call(const std::string &unit_name, const std::string &unit_action, const std::string &data);
void unit_call(const std::string &unit_name, const std::string &unit_action, const std::string &data, std::function<void(const std::shared_ptr<StackFlows::pzmq_data> &)> callback);
std::list<std::string> get_config_file_paths(std::string &base_model_path, std::string &base_model_config_path, const std::string &mode_name);
std::vector<std::string> glob_files(const std::vector<std::string> &patterns);
bool file_exists(const std::string& filePath);
void unicode_to_utf8(unsigned int codepoint, char *output, int *length);
};  // namespace StackFlows
