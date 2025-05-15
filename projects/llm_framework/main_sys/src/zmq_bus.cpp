
/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "zmq_bus.h"

#include "all.h"
#include <stdbool.h>
#include <functional>
#include <cstring>
#include <StackFlowUtil.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#ifdef ENABLE_BSON
#include <bson/bson.h>
#endif

using namespace StackFlows;

void unit_action_match(int com_id, const std::string &json_str);

zmq_bus_com::zmq_bus_com()
{
    exit_flage      = 1;
    err_count       = 0;
    json_str_flage_ = 0;
    reace_event_    = 0;
    raw_msg_len_    = 0;
}

void zmq_bus_com::work(const std::string &zmq_url_format, int port)
{
    _port             = port;
    exit_flage        = 1;
    std::string ports = std::to_string(port);
    std::vector<char> buff(zmq_url_format.length() + ports.length(), 0);
    sprintf((char *)buff.data(), zmq_url_format.c_str(), port);
    _zmq_url = std::string((char *)buff.data());
    SAFE_SETTING("serial_zmq_url", _zmq_url);
    user_chennal_           = std::make_unique<pzmq>(_zmq_url, ZMQ_PULL,
                                                     [this](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data) { this->send_data(data->string()); });
    reace_data_event_thread = std::make_unique<std::thread>(std::bind(&zmq_bus_com::reace_data_event, this));
}

void zmq_bus_com::stop()
{
    exit_flage = 0;
    user_chennal_.reset();
    reace_data_event_thread->join();
}

void zmq_bus_com::on_data(const std::string &data)
{
    unit_action_match(_port, data);
}

void zmq_bus_com::on_raw_data(const std::string &data)
{
    std::string base64_data;
    int ret     = StackFlows::encode_base64(data, base64_data);
    ssize_t pos = json_str_.find("\"data\"");
    if (pos != std::string::npos) {
        ssize_t end_pos = pos + 6;
        int status      = 0;
        for (ssize_t end_pos = pos + 6; end_pos < json_str_.length(); end_pos++) {
            if ((json_str_[end_pos] == '\"') || (json_str_[end_pos] == ',') || (json_str_[end_pos] == '}')) {
                status++;
            }
            if (status == 3) {
                break;
            }
        }
        json_str_.erase(pos, end_pos - pos);
    }
    std::string new_data;
    new_data.resize(json_str_.length() + 10 + base64_data.length());
    memcpy((void *)new_data.data(), json_str_.c_str(), json_str_.length());
    sprintf((char *)(new_data.data() + json_str_.length() - 1), ",\"data\":\"");
    memcpy((void *)(new_data.data() + json_str_.length() + 8), base64_data.c_str(), base64_data.length());
    new_data[json_str_.length() + 8 + base64_data.length()] = '"';
    new_data[json_str_.length() + 9 + base64_data.length()] = '}';
    on_data(new_data);
}

void zmq_bus_com::on_bson_data(const std::string &data)
{
#ifdef ENABLE_BSON
    bson_t *bson = bson_new_from_data((const uint8_t *)data.c_str(), data.length());
    if (!bson) {
        SLOGW("bson is error");
        return ;
    }
    char *json = bson_as_canonical_extended_json(bson, NULL);
    if (!json) {
        SLOGW("bson to json error");
        bson_destroy(bson);
        return ;
    }
    std::string new_data(json);
    on_data(new_data);
    bson_free(json);
    bson_destroy(bson);
#else
    SLOGW("bson not enable");
#endif
}

void zmq_bus_com::send_data(const std::string &data)
{
    // printf("zmq_bus_com::send_data : send:%s\n", data.c_str());
}

void zmq_bus_com::reace_data_event()
{
    // while (exit_flage)
    // {
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    // }
}

void zmq_bus_com::send_data_event()
{
    // while (exit_flage)
    // {
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    // }
}

zmq_bus_com::~zmq_bus_com()
{
    if (exit_flage) {
        stop();
    }
}

unit_data::unit_data()
{
}

void unit_data::init_zmq(const std::string &url)
{
    inference_url           = url;
    user_inference_chennal_ = std::make_unique<pzmq>(inference_url, ZMQ_PUB);
}

void unit_data::send_msg(const std::string &json_str)
{
    user_inference_chennal_->send_data(json_str);
}

unit_data::~unit_data()
{
    user_inference_chennal_.reset();
}

int zmq_bus_publisher_push(const std::string &work_id, const std::string &json_str)
{
    if (work_id.empty()) {
        SLOGW("work_id is empty");
        return -1;
    }
    unit_data *unit_p = NULL;
    SAFE_READING(unit_p, unit_data *, work_id);
    if (unit_p)
        unit_p->send_msg(json_str);
    else {
        SLOGW("zmq_bus_publisher_push failed, not have work_id:%s", work_id.c_str());
        return -1;
    }
    return 0;
}

void *usr_context;

void zmq_com_send(int com_id, const std::string &out_str)
{
    char zmq_push_url[128];
    sprintf(zmq_push_url, zmq_c_format.c_str(), com_id);
    pzmq _zmq(zmq_push_url, ZMQ_PUSH);
    std::string out = out_str + "\n";
    _zmq.send_data(out);
}

void zmq_bus_work()
{
}

void zmq_bus_stop_work()
{
}

void zmq_bus_com::select_json_str(const std::string &json_src, std::function<void(const std::string &)> out_fun)
{
    bool enloop = false;
    std::shared_ptr<std::string> src_str((std::string *)&json_src, [](std::string *pt) {});
    do {
        enloop = false;
        switch (reace_event_) {
            case RAW_NONE: {
                json_str_.reserve(json_str_.length() + src_str->length());
                const char *data = src_str->c_str();
                for (int i = 0; i < src_str->length(); i++) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    if (src_str->length() - i >= 16) {
                        uint8x16_t target_open  = vdupq_n_u8('{');
                        uint8x16_t target_close = vdupq_n_u8('}');
                        uint8x16_t input_vector = vld1q_u8((const uint8_t *)&data[i]);
                        uint8x16_t result_open  = vceqq_u8(input_vector, target_open);
                        uint8x16_t result_close = vceqq_u8(input_vector, target_close);
                        uint8x16_t result_mask  = vorrq_u8(result_open, result_close);
                        __uint128_t jflage;
                        vst1q_u8((uint8_t *)&jflage, result_mask);
                        if (jflage == 0) {
                            json_str_.append(data + i, 16);
                            i += 15;
                            continue;
                        }
                    }
#endif
                    json_str_ += data[i];
                    int last_index = (i == 0) ? 0 : (i - 1);
                    if ((data[i] == '{') && (data[last_index] != '\\')) json_str_flage_++;
                    if ((data[i] == '}') && (data[last_index] != '\\')) json_str_flage_--;
                    if (json_str_flage_ == 0) {
                        if ((json_str_[0] == '{') && (json_str_[json_str_.length() - 1] == '}')) {
                            if ((json_str_.length() > 7) && (json_str_[1] == '\"') && (json_str_[2] == 'R') &&
                                (json_str_[3] == 'A') && (json_str_[4] == 'W') && (json_str_[5] == '\"') &&
                                (json_str_[6] == ':')) {
                                reace_event_ = RAW_JSON;
                                raw_msg_len_ = std::stoi(StackFlows::sample_json_str_get(json_str_, "RAW"));
                                raw_msg_buff_.reserve(raw_msg_len_);
                                if (json_src.length() > i) {
                                    src_str = std::make_shared<std::string>(src_str->substr(i + 1));
                                    enloop  = true;
                                }
                                break;
                            } else if ((json_str_.length() > 7) && (json_str_[1] == '\"') && (json_str_[2] == 'B') &&
                                (json_str_[3] == 'O') && (json_str_[4] == 'N') && (json_str_[5] == '\"') &&
                                (json_str_[6] == ':')) {
                                    reace_event_ = RAW_BSON;
                                    raw_msg_len_ = std::stoi(StackFlows::sample_json_str_get(json_str_, "BON"));
                                    raw_msg_buff_.reserve(raw_msg_len_);
                                    if (json_src.length() > i) {
                                        src_str = std::make_shared<std::string>(src_str->substr(i + 1));
                                        enloop  = true;
                                    }
                                    break;
                            }
                            out_fun(json_str_);
                        }
                        json_str_.clear();
                    }
                    if (json_str_flage_ < 0) {
                        json_str_flage_ = 0;
                        json_str_.clear();
                        throw std::runtime_error("json package error");
                    }
                }
            } break;
            case RAW_JSON: {
                if (raw_msg_len_ > src_str->length()) {
                    raw_msg_buff_.append(src_str->c_str(), src_str->length());
                    raw_msg_len_ -= src_str->length();
                } else {
                    reace_event_ = RAW_NONE;
                    raw_msg_buff_.append(src_str->c_str(), raw_msg_len_);
                    on_raw_data(raw_msg_buff_);
                    raw_msg_buff_.clear();
                    json_str_.clear();
                    if (src_str->length() > raw_msg_len_) {
                        src_str = std::make_shared<std::string>(src_str->substr(raw_msg_len_ + 1));
                        enloop  = true;
                    }
                }
            } break;
            case RAW_BSON: {
                if (raw_msg_len_ > src_str->length()) {
                    raw_msg_buff_.append(src_str->c_str(), src_str->length());
                    raw_msg_len_ -= src_str->length();
                } else {
                    reace_event_ = RAW_NONE;
                    raw_msg_buff_.append(src_str->c_str(), raw_msg_len_);
                    on_bson_data(raw_msg_buff_);
                    raw_msg_buff_.clear();
                    json_str_.clear();
                    if (src_str->length() > raw_msg_len_) {
                        src_str = std::make_shared<std::string>(src_str->substr(raw_msg_len_ + 1));
                        enloop  = true;
                    }
                }
            } break;
            default:
                break;
        }
    } while (enloop);
}