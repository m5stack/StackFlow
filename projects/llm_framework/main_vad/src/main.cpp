/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <thread_safe_list.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"

#define BUFFER_IMPLEMENTATION
#include <stdint.h>
#include "libs/buffer.h"

#include <iostream>

using namespace StackFlows;

int main_exit_flage = 0;

static void __sigint(int iSigNo)
{
    SLOGW("llm_vad will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;

typedef std::function<void(const bool &data)> task_callback_t;

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

class llm_task {
private:
    sherpa_onnx::VadModelConfig mode_config_;
    std::unique_ptr<sherpa_onnx::VoiceActivityDetector> vad_;

public:
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    bool enoutput_;
    bool enstream_;
    bool ensleep_;
    bool printed = false;
    std::atomic_bool superior_flage_;
    std::atomic_bool audio_flage_;
    std::atomic_bool awake_flage_;
    std::string superior_id_;
    task_callback_t out_callback_;
    int awake_delay_       = 50;
    int delay_audio_frame_ = 4;
    buffer_t *pcmdata;
    std::string wake_wav_file_;

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

            CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.model);
            CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.threshold);
            CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.min_silence_duration);
            CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.min_speech_duration);
            CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.window_size);
            CONFIG_AUTO_SET(file_body["mode_param"], sample_rate);
            CONFIG_AUTO_SET(file_body["mode_param"], num_threads);
            CONFIG_AUTO_SET(file_body["mode_param"], provider);

            if (config_body.contains("wake_wav_file"))
                wake_wav_file_ = config_body["wake_wav_file"];
            else if (file_body["mode_param"].contains("wake_wav_file"))
                wake_wav_file_ = file_body["mode_param"]["wake_wav_file"];

            mode_config_.silero_vad.model = base_model + mode_config_.silero_vad.model;
            if (!mode_config_.Validate()) {
                fprintf(stderr, "Errors in config!\n");
                return -1;
            }
            vad_ = std::make_unique<sherpa_onnx::VoiceActivityDetector>(mode_config_);
        } catch (...) {
            SLOGE("config file read false");
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
        buffer_position_set(pcmdata, 0);

        std::vector<float> floatSamples;
        {
            int16_t audio_val;
            while (buffer_read_i16(pcmdata, &audio_val, 1)) {
                float normalizedSample = static_cast<float>(audio_val) / INT16_MAX;
                floatSamples.push_back(normalizedSample);
            }
        }
        buffer_resize(pcmdata, 0);
        count = 0;

        vad_->AcceptWaveform(floatSamples.data(), floatSamples.size());

        if (vad_->IsSpeechDetected() && !printed) {
            printed = true;
            SLOGI("Detected speech!");
            if (out_callback_) {
                out_callback_(true);
            }
        }
        if (!vad_->IsSpeechDetected()) {
            printed = false;
        }
        int32_t sample_rate = 16000;

        while (!vad_->Empty()) {
            const auto &segment = vad_->Front();
            float duration      = segment.samples.size() / static_cast<float>(sample_rate);
            SLOGI("Duration: %.3f seconds", duration);
            // k += 1;
            vad_->Pop();
            if (out_callback_) {
                out_callback_(false);
            }
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
        vad_.reset();
        return true;
    }

    llm_task(const std::string &workid) : audio_flage_(false)
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

class llm_vad : public StackFlow {
private:
    int task_count_;
    std::string audio_url_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    enum { EVENT_LOAD_CONFIG = EVENT_EXPORT + 1, EVENT_TASK_PAUSE };
    llm_vad() : StackFlow("vad")
    {
        task_count_ = 1;
        event_queue_.appendListener(EVENT_TASK_PAUSE, std::bind(&llm_vad::_task_pause, this, std::placeholders::_1));
    }

    void task_output(const std::weak_ptr<llm_task> llm_task_obj_weak,
                     const std::weak_ptr<llm_channel_obj> llm_channel_weak, const bool &data)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::string tmp_msg1;
        const bool *next_data = &data;
        llm_channel->send(llm_task_obj->response_format_, (*next_data), LLM_NO_ERROR);
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
        llm_task_obj->sys_pcm_on_data((*next_data));
    }

    void _task_pause(const std::shared_ptr<void> &arg)
    {
        std::shared_ptr<std::string> work_id = std::static_pointer_cast<std::string>(arg);
        int work_id_num                      = sample_get_work_id_num(*work_id);
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
        event_queue_.enqueue(EVENT_TASK_PAUSE, std::make_shared<std::string>(work_id));
    }

    void task_work(const std::weak_ptr<llm_task> llm_task_obj_weak,
                   const std::weak_ptr<llm_channel_obj> llm_channel_weak)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        if ((!audio_url_.empty()) && (llm_task_obj->audio_flage_ == false)) {
            std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                _llm_task_obj.lock()->sys_pcm_on_data(raw->string());
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
        SLOGI("llm_vad::work:%s", data.c_str());

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
        SLOGI("llm_vad::work:%s", data.c_str());

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
            send("None", "None", error_body, "vad");
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
            send("None", "None", error_body, "vad");
            return -2;
        }
        int ret = llm_task_obj->load_model(config_body);
        if (ret == 0) {
            llm_channel->set_output(llm_task_obj->enoutput_);
            llm_channel->set_stream(llm_task_obj->enstream_);
            llm_task_obj->pause = std::bind(&llm_vad::task_pause, this, work_id, "");
            llm_task_obj->set_output(std::bind(&llm_vad::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("sys") != std::string::npos) {
                    audio_url_                            = unit_call("audio", "cap", "None");
                    std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
                    llm_channel->subscriber(audio_url_,
                                            [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                                                _llm_task_obj.lock()->sys_pcm_on_data(raw->string());
                                            });
                    llm_task_obj->audio_flage_ = true;
                } else if (input.find("vad") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_vad::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if (input.find("kws") != std::string::npos) {
                    llm_task_obj->ensleep_ = true;
                    task_pause(work_id, "");
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_vad::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
            send("None", "None", error_body, "vad");
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_vad::link:%s", data.c_str());
        int ret = 1;
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
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                _llm_task_obj.lock()->sys_pcm_on_data(raw->string());
            });
            llm_task_obj->audio_flage_ = true;
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("kws") != std::string::npos) {
            llm_task_obj->ensleep_ = true;
            ret                    = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_vad::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
        SLOGI("llm_vad::unlink:%s", data.c_str());
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
        if (llm_task_obj->superior_id_ == work_id) {
            llm_task_obj->superior_flage_ = false;
        }
        llm_channel->stop_subscriber_work_id(data);
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
        SLOGI("llm_vad::taskinfo:%s", data.c_str());
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("vad.tasklist", req_body, LLM_NO_ERROR, work_id);
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
            send("vad.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_vad::exit:%s", data.c_str());
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

    ~llm_vad()
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
    llm_vad llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    return 0;
}