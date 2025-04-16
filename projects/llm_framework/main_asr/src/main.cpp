/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "sherpa-ncnn/csrc/recognizer.h"

#include <iostream>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <stdexcept>
#include "../../../../SDK/components/utilities/include/sample_log.h"

#define BUFFER_IMPLEMENTATION
#include <stdbool.h>
#include <stdint.h>
#include "libs/buffer.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_asr will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;

typedef std::function<void(const std::string &data, bool finish)> task_callback_t;

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

class llm_task {
private:
    sherpa_ncnn::RecognizerConfig mode_config_;
    std::unique_ptr<sherpa_ncnn::Recognizer> recognizer_;
    std::unique_ptr<sherpa_ncnn::Stream> recognizer_stream_;

public:
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    bool enoutput_;
    bool enstream_;
    bool ensleep_;
    task_callback_t out_callback_;
    std::atomic_bool audio_flage_;
    std::atomic_bool awake_flage_;
    int awake_delay_       = 50;
    int delay_audio_frame_ = 100;
    buffer_t *pcmdata;

    std::function<void(void)> pause;

    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            model_           = config_body.at("model");
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");
            if (config_body.contains("input")) {
                if (config_body["input"].is_string()) {
                    inputs_.push_back(config_body["input"].get<std::string>());
                } else if (config_body["input"].is_array()) {
                    for (auto _in : config_body["input"]) {
                        inputs_.push_back(_in.get<std::string>());
                    }
                }
            }
        } catch (...) {
            SLOGE("setup config_body error");
            return true;
        }
        enstream_ = response_format_.find("stream") == std::string::npos ? false : true;
        return false;
    }

    int load_model(const nlohmann::json &config_body)
    {
        if (parse_config(config_body)) {
            return -1;
        }
        nlohmann::json file_body;
        std::list<std::string> config_file_paths =
            get_config_file_paths(base_model_path_, base_model_config_path_, model_);
        try {
            for (auto file_name : config_file_paths) {
                std::ifstream config_file(file_name);
                if (!config_file.is_open()) {
                    SLOGW("config file :%s miss", file_name.c_str());
                    continue;
                }
                SLOGI("config file :%s read", file_name.c_str());
                config_file >> file_body;
                config_file.close();
                break;
            }
            if (file_body.empty()) {
                SLOGE("all config file miss");
                return -2;
            }
            std::string base_model = base_model_path_ + model_ + "/";
            SLOGI("base_model %s", base_model.c_str());

            CONFIG_AUTO_SET(file_body["mode_param"], feat_config.sampling_rate);
            CONFIG_AUTO_SET(file_body["mode_param"], feat_config.feature_dim);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.encoder_param);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.encoder_bin);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.decoder_param);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.decoder_bin);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.joiner_param);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.joiner_bin);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.tokens);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.encoder_opt.num_threads);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.decoder_opt.num_threads);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.joiner_opt.num_threads);
            CONFIG_AUTO_SET(file_body["mode_param"], decoder_config.method);
            CONFIG_AUTO_SET(file_body["mode_param"], decoder_config.num_active_paths);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.must_contain_nonsilence);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.min_trailing_silence);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.min_utterance_length);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.must_contain_nonsilence);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.min_trailing_silence);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.min_utterance_length);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.must_contain_nonsilence);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.min_trailing_silence);
            CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.min_utterance_length);
            CONFIG_AUTO_SET(file_body["mode_param"], enable_endpoint);
            CONFIG_AUTO_SET(file_body["mode_param"], hotwords_file);
            CONFIG_AUTO_SET(file_body["mode_param"], hotwords_score);
            if (config_body.contains("awake_delay"))
                awake_delay_ = config_body["awake_delay"].get<int>();
            else if (file_body["mode_param"].contains("awake_delay"))
                awake_delay_ = file_body["mode_param"]["awake_delay"];
            if (config_body.contains("rule1")) {
                mode_config_.endpoint_config.rule1.min_trailing_silence = config_body["rule1"].get<float>();
                mode_config_.endpoint_config.rule1.must_contain_nonsilence =
                    (mode_config_.endpoint_config.rule1.min_trailing_silence == 0.0f) ? false : true;
            }
            if (config_body.contains("rule2")) {
                mode_config_.endpoint_config.rule2.min_trailing_silence = config_body["rule2"].get<float>();
                mode_config_.endpoint_config.rule2.must_contain_nonsilence =
                    (mode_config_.endpoint_config.rule2.min_trailing_silence == 0.0f) ? false : true;
            }
            if (config_body.contains("rule3")) {
                mode_config_.endpoint_config.rule3.min_utterance_length = config_body["rule3"].get<float>();
                mode_config_.endpoint_config.rule3.must_contain_nonsilence =
                    (mode_config_.endpoint_config.rule3.min_utterance_length == 0.0f) ? false : true;
            }

            mode_config_.model_config.tokens        = base_model + mode_config_.model_config.tokens;
            mode_config_.model_config.encoder_param = base_model + mode_config_.model_config.encoder_param;
            mode_config_.model_config.encoder_bin   = base_model + mode_config_.model_config.encoder_bin;
            mode_config_.model_config.decoder_param = base_model + mode_config_.model_config.decoder_param;
            mode_config_.model_config.decoder_bin   = base_model + mode_config_.model_config.decoder_bin;
            mode_config_.model_config.joiner_param  = base_model + mode_config_.model_config.joiner_param;
            mode_config_.model_config.joiner_bin    = base_model + mode_config_.model_config.joiner_bin;
            recognizer_                             = std::make_unique<sherpa_ncnn::Recognizer>(mode_config_);
        } catch (...) {
            SLOGE("config false");
            return -3;
        }
        return 0;
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

    void sys_pcm_on_data(const std::string &raw)
    {
        static int count = 0;
        if (count < delay_audio_frame_) {
            buffer_write_char(pcmdata, raw.c_str(), raw.length());
            count++;
            return;
        }
        buffer_write_char(pcmdata, raw.c_str(), raw.length());
        buffer_position_set(pcmdata, 0);
        count = 0;
        std::vector<float> floatSamples;
        {
            int16_t audio_val;
            while (buffer_read_u16(pcmdata, (unsigned short *)&audio_val, 1)) {
                float normalizedSample = (float)audio_val / INT16_MAX;
                floatSamples.push_back(normalizedSample);
            }
        }
        buffer_position_set(pcmdata, 0);
        if (awake_flage_ && recognizer_stream_) {
            recognizer_stream_.reset();
            awake_flage_ = false;
        }
        if (!recognizer_stream_) {
            recognizer_stream_ = recognizer_->CreateStream();
        }
        recognizer_stream_->AcceptWaveform(mode_config_.feat_config.sampling_rate, floatSamples.data(),
                                           floatSamples.size());
        while (recognizer_->IsReady(recognizer_stream_.get())) {
            recognizer_->DecodeStream(recognizer_stream_.get());
        }
        std::string text = recognizer_->GetResult(recognizer_stream_.get()).text;
        std::string lower_text;
        lower_text.resize(text.size());
        std::transform(text.begin(), text.end(), lower_text.begin(), [](const char c) { return std::tolower(c); });
        if ((!lower_text.empty()) && out_callback_) out_callback_(lower_text, false);
        bool is_endpoint = recognizer_->IsEndpoint(recognizer_stream_.get());
        if (is_endpoint) {
            std::cout << "asr have a is_endpoint \n";
            recognizer_stream_->Finalize();
            if ((!lower_text.empty()) && out_callback_) {
                out_callback_(lower_text, true);
            }
            recognizer_stream_.reset();
            if (ensleep_) {
                if (pause) pause();
            }
        }
    }

    void kws_awake()
    {
        awake_flage_ = true;
    }

    bool delete_model()
    {
        recognizer_.reset();
        return true;
    }

    llm_task(const std::string &workid)
    {
        ensleep_     = false;
        awake_flage_ = false;
        pcmdata      = buffer_create();
    }

    void start()
    {
    }

    void stop()
    {
    }

    ~llm_task()
    {
        stop();
        buffer_destroy(pcmdata);
    }
};

#undef CONFIG_AUTO_SET

class llm_asr : public StackFlow {
public:
    enum { EVENT_LOAD_CONFIG = EVENT_EXPORT + 1, EVENT_TASK_PAUSE };

private:
    int task_count_;
    std::string audio_url_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_asr() : StackFlow("asr")
    {
        task_count_ = 1;
        event_queue_.appendListener(
            EVENT_TASK_PAUSE, std::bind(&llm_asr::_task_pause, this, std::placeholders::_1, std::placeholders::_2));
    }

    void task_output(const std::weak_ptr<llm_task> llm_task_obj_weak,
                     const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &data, bool finish)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::string tmp_msg1;
        const std::string *next_data = &data;
        if (finish) {
            tmp_msg1  = data + ".";
            next_data = &tmp_msg1;
        }
        if (llm_channel->enstream_) {
            static int count = 0;
            nlohmann::json data_body;
            data_body["index"]  = count++;
            data_body["delta"]  = (*next_data);
            data_body["finish"] = finish;
            if (finish) count = 0;
            SLOGI("send stream:%s", next_data->c_str());
            llm_channel->send(llm_task_obj->response_format_, data_body, LLM_NO_ERROR);
        } else if (finish) {
            SLOGI("send utf-8:%s", next_data->c_str());
            llm_channel->send(llm_task_obj->response_format_, (*next_data), LLM_NO_ERROR);
        }
    }

    int decode_wav(const std::string &in, std::string &out)
    {
        int post = 0;
        if (in.length() > 10)
            for (int i = 0; i < in.length() - 4; i++) {
                if ((in[i] == 'd') && (in[i + 1] == 'a') && (in[i + 2] == 't') && (in[i + 3] == 'a')) {
                    post = i + 8;
                    break;
                }
            }
        if (post > 0) {
            out = std::string((char *)(in.c_str() + post), in.length() - post);
            return in.length() - post;
        } else {
            return 0;
        }
        return 0;
    }

    int decode_mp3(const std::string &in, std::string &out)
    {
        return 0;
    }

    void task_user_data(const std::weak_ptr<llm_task> llm_task_obj_weak,
                        const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                        const std::string &data)
    {
        nlohmann::json error_body;
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            error_body["code"]    = -11;
            error_body["message"] = "Model run failed.";
            send("None", "None", error_body, unit_name_);
            return;
        }
        std::string tmp_msg1;
        const std::string *next_data = &data;
        int ret;
        if (object.find("stream") != std::string::npos) {
            static std::unordered_map<int, std::string> stream_buff;
            try {
                if (decode_stream(data, tmp_msg1, stream_buff)) {
                    return;
                };
            } catch (...) {
                stream_buff.clear();
                error_body["code"]    = -25;
                error_body["message"] = "Stream data index error.";
                send("None", "None", error_body, unit_name_);
                return;
            }
            next_data = &tmp_msg1;
        }
        std::string tmp_msg2;
        if (object.find("base64") != std::string::npos) {
            ret = decode_base64((*next_data), tmp_msg2);
            if (ret == -1) {
                error_body["code"]    = -23;
                error_body["message"] = "Base64 decoding error.";
                send("None", "None", error_body, unit_name_);
                return;
            }
            next_data = &tmp_msg2;
        }
        std::string tmp_msg3;
        if (object.find("wav") != std::string::npos) {
            ret = decode_wav((*next_data), tmp_msg3);
            if (!ret) {
                return;
            }
            next_data = &tmp_msg3;
        }
        std::string tmp_msg4;
        if (object.find("mp3") != std::string::npos) {
            ret = decode_mp3((*next_data), tmp_msg4);
            if (!ret) {
                return;
            }
            next_data = &tmp_msg4;
        }
        llm_task_obj->sys_pcm_on_data((*next_data));
    }

    void _task_pause(const std::string &work_id, const std::string &data)
    {
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            return;
        }
        auto llm_task_obj = llm_task_[work_id_num];
        auto llm_channel  = get_channel(work_id_num);
        if (llm_task_obj->audio_flage_) {
            if (!audio_url_.empty()) llm_channel->stop_subscriber(audio_url_);
            llm_task_obj->audio_flage_ = false;
        }
    }

    void task_pause(const std::string &work_id, const std::string &data)
    {
        event_queue_.enqueue(EVENT_TASK_PAUSE, work_id, "");
    }

    void task_work(const std::weak_ptr<llm_task> llm_task_obj_weak,
                   const std::weak_ptr<llm_channel_obj> llm_channel_weak)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        llm_task_obj->kws_awake();
        if ((!audio_url_.empty()) && (llm_task_obj->audio_flage_ == false)) {
            std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::string &raw) {
                _llm_task_obj.lock()->sys_pcm_on_data(raw);
            });
            llm_task_obj->audio_flage_ = true;
        }
    }

    void kws_awake(const std::weak_ptr<llm_task> llm_task_obj_weak,
                   const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                   const std::string &data)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(llm_task_obj->awake_delay_));
        task_work(llm_task_obj, llm_channel);
    }

    void work(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_asr::work:%s", data.c_str());

        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        task_work(llm_task_[work_id_num], get_channel(work_id_num));
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    void pause(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_asr::work:%s", data.c_str());

        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        task_pause(work_id, "");
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        if ((llm_task_channel_.size() - 1) == task_count_) {
            error_body["code"]    = -21;
            error_body["message"] = "task full";
            send("None", "None", error_body, "asr");
            return -1;
        }

        int work_id_num   = sample_get_work_id_num(work_id);
        auto llm_channel  = get_channel(work_id);
        auto llm_task_obj = std::make_shared<llm_task>(work_id);

        nlohmann::json config_body;
        try {
            config_body = nlohmann::json::parse(data);
        } catch (...) {
            SLOGE("setup json format error.");
            error_body["code"]    = -2;
            error_body["message"] = "json format error.";
            send("None", "None", error_body, "kws");
            return -2;
        }
        int ret = llm_task_obj->load_model(config_body);
        if (ret == 0) {
            llm_channel->set_output(llm_task_obj->enoutput_);
            llm_channel->set_stream(llm_task_obj->enstream_);
            llm_task_obj->pause = std::bind(&llm_asr::task_pause, this, work_id, "");
            llm_task_obj->set_output(std::bind(&llm_asr::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("sys") != std::string::npos) {
                    audio_url_                            = unit_call("audio", "cap", input);
                    std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
                    llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::string &raw) {
                        _llm_task_obj.lock()->sys_pcm_on_data(raw);
                    });
                    llm_task_obj->audio_flage_ = true;
                } else if (input.find("asr") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_asr::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if (input.find("kws") != std::string::npos) {
                    llm_task_obj->ensleep_ = true;
                    task_pause(work_id, "");
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_asr::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                }
            }
            llm_task_[work_id_num] = llm_task_obj;
            SLOGI("load_mode success");
            send("None", "None", LLM_NO_ERROR, work_id);
            return 0;
        } else {
            SLOGE("load_mode Failed");
            error_body["code"]    = -5;
            error_body["message"] = "Model loading failed.";
            send("None", "None", error_body, "asr");
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_asr::link:%s", data.c_str());
        int ret = 0;
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        auto llm_channel  = get_channel(work_id);
        auto llm_task_obj = llm_task_[work_id_num];
        if (data.find("sys") != std::string::npos) {
            if (audio_url_.empty()) audio_url_ = unit_call("audio", "cap", data);
            std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::string &raw) {
                _llm_task_obj.lock()->sys_pcm_on_data(raw);
            });
            llm_task_obj->audio_flage_ = true;
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("kws") != std::string::npos) {
            llm_task_obj->ensleep_ = true;
            ret                    = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_asr::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
                                             std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->inputs_.push_back(data);
        }
        if (ret) {
            error_body["code"]    = -20;
            error_body["message"] = "link false";
            send("None", "None", error_body, work_id);
            return;
        } else {
            send("None", "None", LLM_NO_ERROR, work_id);
        }
    }

    void unlink(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_asr::unlink:%s", data.c_str());
        int ret = 0;
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        auto llm_channel = get_channel(work_id);
        llm_channel->stop_subscriber_work_id(data);
        auto llm_task_obj = llm_task_[work_id_num];
        for (auto it = llm_task_obj->inputs_.begin(); it != llm_task_obj->inputs_.end();) {
            if (*it == data) {
                it = llm_task_obj->inputs_.erase(it);
            } else {
                ++it;
            }
        }
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    void taskinfo(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_asr::taskinfo:%s", data.c_str());

        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("asr.tasklist", req_body, LLM_NO_ERROR, work_id);
        } else {
            if (llm_task_.find(work_id_num) == llm_task_.end()) {
                req_body["code"]    = -6;
                req_body["message"] = "Unit Does Not Exist";
                send("None", "None", req_body, work_id);
                return;
            }
            auto llm_task_obj           = llm_task_[work_id_num];
            req_body["model"]           = llm_task_obj->model_;
            req_body["response_format"] = llm_task_obj->response_format_;
            req_body["enoutput"]        = llm_task_obj->enoutput_;
            req_body["inputs"]          = llm_task_obj->inputs_;
            send("asr.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_asr::exit:%s", data.c_str());
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return -1;
        }
        llm_task_[work_id_num]->stop();
        auto llm_channel = get_channel(work_id_num);
        llm_channel->stop_subscriber("");
        if (llm_task_[work_id_num]->audio_flage_) {
            unit_call("audio", "cap_stop", "None");
        }
        llm_task_.erase(work_id_num);
        send("None", "None", LLM_NO_ERROR, work_id);
        return 0;
    }

    ~llm_asr()
    {
        while (1) {
            auto iteam = llm_task_.begin();
            if (iteam == llm_task_.end()) {
                break;
            }
            iteam->second->stop();
            if (iteam->second->audio_flage_) {
                unit_call("audio", "cap_stop", "None");
            }
            get_channel(iteam->first)->stop_subscriber("");
            iteam->second.reset();
            llm_task_.erase(iteam->first);
        }
    }
};

int main(int argc, char *argv[])
{
    signal(SIGTERM, __sigint);
    signal(SIGINT, __sigint);
    mkdir("/tmp/llm", 0777);
    llm_asr llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}