/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlowUtil.h"
#include <vector>

std::string StackFlows::sample_json_str_get(const std::string &json_str, const std::string &json_key) {
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

int StackFlows::sample_get_work_id_num(const std::string &work_id) {
    int a = work_id.find(".");
    if ((a == std::string::npos) || (a == work_id.length() - 1)) {
        return WORK_ID_NONE;
    }
    return std::stoi(work_id.substr(a + 1));
}

std::string StackFlows::sample_get_work_id_name(const std::string &work_id) {
    int a = work_id.find(".");
    if (a == std::string::npos) {
        return work_id;
    } else {
        return work_id.substr(0, a);
    }
}

std::string StackFlows::sample_get_work_id(int work_id_num, const std::string &unit_name) {
    return unit_name + "." + std::to_string(work_id_num);
}

std::string StackFlows::sample_escapeString(const std::string &input) {
    std::string escaped;
    for (char c : input) {
        switch (c) {
            case '\n':
                escaped += "\\n";
                break;
            case '\t':
                escaped += "\\t";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\"':
                escaped += "\\\"";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\b':
                escaped += "\\b";
                break;
            default:
                escaped += c;
                break;
        }
    }
    return escaped;
}

std::string StackFlows::sample_unescapeString(const std::string &input) {
    std::string unescaped;
    for (size_t i = 0; i < input.length(); ++i) {
        if (input[i] == '\\' && i + 1 < input.length()) {
            switch (input[i + 1]) {
                case 'n':
                    unescaped += '\n';
                    ++i;
                    break;
                case 't':
                    unescaped += '\t';
                    ++i;
                    break;
                case '\\':
                    unescaped += '\\';
                    ++i;
                    break;
                case '\"':
                    unescaped += '\"';
                    ++i;
                    break;
                case 'r':
                    unescaped += '\r';
                    ++i;
                    break;
                case 'b':
                    unescaped += '\b';
                    ++i;
                    break;
                default:
                    unescaped += input[i];
                    break;
            }
        } else {
            unescaped += input[i];
        }
    }
    return unescaped;
}

bool StackFlows::decode_stream(const std::string &in, std::string &out,
                               std::unordered_map<int, std::string> &stream_buff) {
    try {
        int index          = std::stoi(StackFlows::sample_json_str_get(in, "index"));
        std::string finish = StackFlows::sample_json_str_get(in, "finish");
        std::string delta  = StackFlows::sample_json_str_get(in, "delta");
        stream_buff[index] = delta;
        if (finish == "true") {
            for (size_t i = 0; i < stream_buff.size(); i++) {
                out += stream_buff.at(i);
            }
            stream_buff.clear();
            return false;
        } else if (finish != "false") {
            throw true;
        }
    } catch (...) {
        stream_buff.clear();
    }
    return true;
}

#define BASE64_ENCODE_OUT_SIZE(s) (((s) + 2) / 3 * 4)
#define BASE64_DECODE_OUT_SIZE(s) (((s)) / 4 * 3)
#include <stdio.h>
/* BASE 64 encode table */
static const char base64en[] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
};

#define BASE64_PAD     '='
#define BASE64DE_FIRST '+'
#define BASE64DE_LAST  'z'
/* ASCII order for BASE 64 decode, -1 in unused character */
static const signed char base64de[] = {
    /* '+', ',', '-', '.', '/', '0', '1', '2', */
    62,
    -1,
    -1,
    -1,
    63,
    52,
    53,
    54,

    /* '3', '4', '5', '6', '7', '8', '9', ':', */
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    -1,

    /* ';', '<', '=', '>', '?', '@', 'A', 'B', */
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    0,
    1,

    /* 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', */
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,

    /* 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', */
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,

    /* 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', */
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,

    /* '[', '\', ']', '^', '_', '`', 'a', 'b', */
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    26,
    27,

    /* 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', */
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,

    /* 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', */
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,

    /* 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', */
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
};

static int base64_encode(const unsigned char *in, unsigned int inlen, char *out) {
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

static int base64_decode(const char *in, unsigned int inlen, unsigned char *out) {
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

int StackFlows::decode_base64(const std::string &in, std::string &out) {
    out.resize(BASE64_DECODE_OUT_SIZE(in.length()));
    return base64_decode((const char *)in.c_str(), in.length(), (unsigned char *)out.data());
}

int StackFlows::encode_base64(const std::string &in, std::string &out) {
    out.resize(BASE64_ENCODE_OUT_SIZE(in.length()));
    return base64_encode((const unsigned char *)in.c_str(), in.length(), (char *)out.data());
}