/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlowUtil.h"
#include <vector>
#include <glob.h>
#include <fstream>
#include "pzmq.hpp"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// #include <iconv.h>

std::string StackFlows::sample_json_str_get(const std::string &json_str, const std::string &json_key)
{
    std::string key_val;
    std::string format_val;
    // SLOGD("json_str: %s json_key:%s\n", json_str.c_str(), json_key.c_str());
    std::string find_key = "\"" + json_key + "\"";
    int subs_start       = json_str.find(find_key);
    if (subs_start == std::string::npos) {
        return key_val;
    }
    int status    = 0;
    char last_c   = '\0';
    int obj_flage = 0;
    for (auto c : json_str.substr(subs_start + find_key.length())) {
        switch (status) {
            case 0: {
                switch (c) {
                    case '"': {
                        status = 100;
                    } break;
                    case '{': {
                        key_val.push_back(c);
                        obj_flage = 1;
                        status    = 10;
                    } break;
                    case ':':
                        obj_flage = 1;
                        break;
                    case ',':
                    case '}': {
                        obj_flage = 0;
                        status    = -1;
                    } break;
                    case ' ':
                        break;
                    default: {
                        if (obj_flage) {
                            key_val.push_back(c);
                        }
                    } break;
                }
            } break;
            case 10: {
                key_val.push_back(c);
                if (c == '{') {
                    obj_flage++;
                }
                if (c == '}') {
                    obj_flage--;
                }
                if (obj_flage == 0) {
                    if (!key_val.empty()) {
                        status = -1;
                    }
                }
            } break;
            case 100: {
                if ((c == '"') && (last_c != '\\')) {
                    obj_flage = 0;
                    status    = -1;
                } else {
                    key_val.push_back(c);
                }
            } break;
            default:
                break;
        }
        last_c = c;
    }
    if (obj_flage != 0) {
        key_val.clear();
    }
    // SLOGD("key_val:%s\n", key_val.c_str());
    return key_val;
}

int StackFlows::sample_get_work_id_num(const std::string &work_id)
{
    int a = work_id.find(".");
    if ((a == std::string::npos) || (a == work_id.length() - 1)) {
        return WORK_ID_NONE;
    }
    return std::stoi(work_id.substr(a + 1));
}

std::string StackFlows::sample_get_work_id_name(const std::string &work_id)
{
    int a = work_id.find(".");
    if (a == std::string::npos) {
        return work_id;
    } else {
        return work_id.substr(0, a);
    }
}

std::string StackFlows::sample_get_work_id(int work_id_num, const std::string &unit_name)
{
    return unit_name + "." + std::to_string(work_id_num);
}
// clang-format off
std::string StackFlows::sample_escapeString(const std::string &input)
{
    std::string escaped;
    for (char c : input) {
        switch (c) {
            case '\n':escaped += "\\n" ;break;
            case '\t':escaped += "\\t" ;break;
            case '\\':escaped += "\\\\";break;
            case '\"':escaped += "\\\"";break;
            case '\r':escaped += "\\r" ;break;
            case '\b':escaped += "\\b" ;break;
            default  :escaped += c     ;break;
        }
    }
    return escaped;
}

void StackFlows::unicode_to_utf8(unsigned int codepoint, char *output, int *length) {
    if (codepoint <= 0x7F) {
        output[0] = codepoint & 0x7F;
        *length = 1;
    } else if (codepoint <= 0x7FF) {
        output[0] = 0xC0 | ((codepoint >> 6) & 0x1F);
        output[1] = 0x80 | (codepoint & 0x3F);
        *length = 2;
    } else if (codepoint <= 0xFFFF) {
        output[0] = 0xE0 | ((codepoint >> 12) & 0x0F);
        output[1] = 0x80 | ((codepoint >> 6) & 0x3F);
        output[2] = 0x80 | (codepoint & 0x3F);
        *length = 3;
    } else if (codepoint <= 0x10FFFF) {
        output[0] = 0xF0 | ((codepoint >> 18) & 0x07);
        output[1] = 0x80 | ((codepoint >> 12) & 0x3F);
        output[2] = 0x80 | ((codepoint >> 6) & 0x3F);
        output[3] = 0x80 | (codepoint & 0x3F);
        *length = 4;
    } else {
        *length = 0;
    }
}

std::string StackFlows::sample_unescapeString(const std::string &input, bool ucs2)
{
    std::string unescaped;
    unescaped.reserve(input.length());
    const char *itc = input.c_str();
    const char *itd = itc + input.length();
    while (itc != itd){
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if ((itd - itc) >= 16) {
            const uint8_t *srcstr = (const uint8_t *)itc;
            uint8x16_t target_open  = vdupq_n_u8('\\');
            uint8x16_t input_vector = vld1q_u8(srcstr);
            uint8x16_t result_open  = vceqq_u8(input_vector, target_open);
            __uint128_t jflage;
            vst1q_u8((uint8_t *)&jflage, result_open);
            if (jflage == 0) {
                int pos = unescaped.size();
                unescaped.resize(pos + 16);
                memcpy((void*)(unescaped.c_str() + pos), srcstr, 16);
                itc += 16;
                continue;
            }
        }
#endif
        if((*itc == '\\') && (++itc != itd)){
            switch (*itc)
            {
                case 'n' :unescaped += '\n';itc++;break;
                case 't' :unescaped += '\t';itc++;break;
                case '\\':unescaped += '\\';itc++;break;
                case '\"':unescaped += '\"';itc++;break;
                case 'r' :unescaped += '\r';itc++;break;
                case 'b' :unescaped += '\b';itc++;break;
                case 'u' :{
                    itc++;
                    if(ucs2) {
                        unescaped += "\\u";
                        break;
                    }
                    unsigned int codepoint = 0;
                    auto itcb = itc;
                    do {
                        if((itd - itc) < 4) {
                            codepoint = -1;
                            break;
                        }
                        for (int i = 0; i < 4; i++) {
                            char c = *itc++;
                            codepoint <<= 4;
                            switch (c) {
                                case '0' ... '9' : codepoint |= (c - '0'); break;
                                case 'A' ... 'F' : codepoint |= (c - 'A' + 10); break;
                                case 'a' ... 'f' : codepoint |= (c - 'a' + 10); break;
                                default: codepoint = -1; i += 4 ; break;
                            }
                        }
                    } while (codepoint >= 0xD800 && codepoint <= 0xDBFF);
                    if(codepoint == -1) {
                        itc = itcb;
                        unescaped += "\\u";
                        break;
                    }
                    char buff[4];
                    int len = 0;
                    StackFlows::unicode_to_utf8(codepoint, buff, &len);
                    if(len != 0) {
                        for (int i = 0; i < len; i++) {
                            unescaped += buff[i];
                        }
                    } else {
                        itc = itcb;
                        unescaped += "\\u";
                    }
                }
                break;
            default:
                unescaped += *itc++;
                break;
            }
        }
        else {
            unescaped += *itc++;
        }
    }
    return unescaped;
}
// clang-format on
bool StackFlows::decode_stream(const std::string &in, std::string &out,
                               std::unordered_map<int, std::string> &stream_buff)
{
    int index          = std::stoi(StackFlows::sample_json_str_get(in, "index"));
    std::string finish = StackFlows::sample_json_str_get(in, "finish");
    stream_buff[index] = StackFlows::sample_json_str_get(in, "delta");
    // sample find flage: false:true
    if (finish.find("f") == std::string::npos) {
        for (size_t i = 0; i < stream_buff.size(); i++) {
            out += stream_buff.at(i);
        }
        stream_buff.clear();
        return false;
    }
    return true;
}

#define BASE64_ENCODE_OUT_SIZE(s) (((s) + 2) / 3 * 4)
#define BASE64_DECODE_OUT_SIZE(s) (((s)) / 4 * 3)
#include <stdio.h>
/* BASE 64 encode table */
static const char base64en[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#define BASE64_PAD     '='
#define BASE64DE_FIRST '+'
#define BASE64DE_LAST  'z'
/* ASCII order for BASE 64 decode, -1 in unused character */
static const signed char base64de[] = {
    62, -1, -1, -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,
    5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, -1,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
};

static int base64_encode(const unsigned char *in, unsigned int inlen, char *out)
{
    unsigned int i = 0, j = 0;
    for (; i < inlen; i++) {
        int s = i % 3;
        switch (s) {
            case 0:
                out[j++] = base64en[(in[i] >> 2) & 0x3F];
                continue;
            case 1:
                out[j++] = base64en[((in[i - 1] & 0x3) << 4) + ((in[i] >> 4) & 0xF)];
                continue;
            case 2:
                out[j++] = base64en[((in[i - 1] & 0xF) << 2) + ((in[i] >> 6) & 0x3)];
                out[j++] = base64en[in[i] & 0x3F];
        }
    }
    /* move back */
    i -= 1;
    /* check the last and add padding */
    if ((i % 3) == 0) {
        out[j++] = base64en[(in[i] & 0x3) << 4];
        out[j++] = BASE64_PAD;
        out[j++] = BASE64_PAD;
    } else if ((i % 3) == 1) {
        out[j++] = base64en[(in[i] & 0xF) << 2];
        out[j++] = BASE64_PAD;
    }
    return j;
}

static int base64_decode(const char *in, unsigned int inlen, unsigned char *out)
{
    unsigned int i = 0, j = 0;
    for (; i < inlen; i++) {
        int c;
        int s = i % 4;
        if (in[i] == '=') return j;
        if (in[i] < BASE64DE_FIRST || in[i] > BASE64DE_LAST || (c = base64de[in[i] - BASE64DE_FIRST]) == -1) return -1;
        switch (s) {
            case 0:
                out[j] = ((unsigned int)c << 2) & 0xFF;
                continue;
            case 1:
                out[j++] += ((unsigned int)c >> 4) & 0x3;
                /* if not last char with padding */
                if (i < (inlen - 3) || in[inlen - 2] != '=') out[j] = ((unsigned int)c & 0xF) << 4;
                continue;
            case 2:
                out[j++] += ((unsigned int)c >> 2) & 0xF;
                /* if not last char with padding */
                if (i < (inlen - 2) || in[inlen - 1] != '=') out[j] = ((unsigned int)c & 0x3) << 6;
                continue;
            case 3:
                out[j++] += (unsigned char)c;
        }
    }

    return j;
}

int StackFlows::decode_base64(const std::string &in, std::string &out)
{
    int out_size = BASE64_DECODE_OUT_SIZE(in.length());
    out.resize(out_size);
    int ret = base64_decode((const char *)in.c_str(), in.length(), (unsigned char *)out.data());
    if ((ret > 0) && (ret != out_size)) out.erase(ret);
    return ret;
}

int StackFlows::encode_base64(const std::string &in, std::string &out)
{
    int out_size = BASE64_ENCODE_OUT_SIZE(in.length());
    out.resize(out_size);
    int ret = base64_encode((const unsigned char *)in.c_str(), in.length(), (char *)out.data());
    if ((ret > 0) && (ret != out_size)) out.erase(ret);
    return ret;
}

std::string StackFlows::unit_call(const std::string &unit_name, const std::string &unit_action, const std::string &data)
{
    std::string value;
    pzmq _call(unit_name);
    _call.call_rpc_action(unit_action, data, [&value](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) { value = raw->string(); });
    return value;
}

std::list<std::string> StackFlows::get_config_file_paths(std::string &base_model_path,
                                                         std::string &base_model_config_path,
                                                         const std::string &mode_name)
{
    if (base_model_path.empty()) {
        base_model_path = sample_unescapeString(unit_call("sys", "sql_select", "config_base_mode_path"));
        if (base_model_path.empty()) {
            base_model_path = "/opt/m5stack/data/";
        }
    }
    if (base_model_config_path.empty()) {
        base_model_config_path = sample_unescapeString(unit_call("sys", "sql_select", "config_base_mode_config_path"));
        if (base_model_config_path.empty()) {
            base_model_config_path = "/opt/m5stack/etc/";
        }
    }
    std::string config_model_d = sample_unescapeString(unit_call("sys", "sql_select", "config_model_d"));
    if (config_model_d.empty()) config_model_d = "/opt/m5stack/data/models/";

    std::list<std::string> config_file_paths;
    config_file_paths.push_back(std::string("./") + mode_name + ".json");
    config_file_paths.push_back(std::string("./mode_") + mode_name + ".json");
    config_file_paths.push_back(base_model_path + mode_name + std::string("/") + mode_name + ".json");
    config_file_paths.push_back(base_model_path + mode_name + std::string("/") + std::string("./mode_") + mode_name +
                                ".json");
    config_file_paths.push_back(config_model_d + mode_name + ".json");
    config_file_paths.push_back(config_model_d + std::string("./mode_") + mode_name + ".json");
    config_file_paths.push_back(base_model_config_path + mode_name + ".json");
    config_file_paths.push_back(base_model_config_path + std::string("./mode_") + mode_name + ".json");
    config_file_paths.push_back(base_model_path + std::string("../share/") + mode_name + ".json");
    config_file_paths.push_back(base_model_path + std::string("../share/") + std::string("./mode_") + mode_name +
                                ".json");
    return config_file_paths;
}

std::vector<std::string> StackFlows::glob_files(const std::vector<std::string> &patterns)
{
    std::vector<std::string> files;
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    for (const auto &pattern : patterns) {
        int ret = glob(pattern.c_str(), GLOB_TILDE | GLOB_BRACE, nullptr, &glob_result);
        if (ret != 0) {
            // if (ret == GLOB_NOMATCH) {
            //     std::cerr << "No files matched for pattern: " << pattern << std::endl;
            // } else {
            //     std::cerr << "glob() failed with error code: " << ret << std::endl;
            // }
            continue;
        }
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            files.push_back(glob_result.gl_pathv[i]);
        }
    }
    globfree(&glob_result);
    return files;
}

bool StackFlows::file_exists(const std::string &filePath)
{
    std::ifstream file(filePath);
    return file.good();
}