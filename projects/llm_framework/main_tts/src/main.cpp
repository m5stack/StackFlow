/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "utils.h"
#include "SynthesizerTrn.h"

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <stdexcept>
#include "../../../../SDK/components/utilities/include/sample_log.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_sys will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;

struct SynthesizerTrn_config {
    int spacker_role    = 0;
    float spacker_speed = 1.0;
    std::string ttsModelName;
};

typedef std::function<void(const std::string &data, bool finish)> task_callback_t;

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

class llm_task {
private:
    float *dataW = NULL;
    int modelSize;

public:
    std::unique_ptr<SynthesizerTrn> synthesizer_;
    SynthesizerTrn_config mode_config_;
    std::vector<std::string> inputs_;
    bool enoutput_;
    bool enstream_;
    std::atomic_bool superior_flage_;
    std::string superior_id_;
    std::string model_;
    std::string response_format_;
    task_callback_t out_callback_;
    int awake_delay_ = 1000;

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

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
        enstream_ = (response_format_.find("stream") != std::string::npos);
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
        // Compatible operation
        if (model_ == "single_speaker_english_fast")
            config_file_paths =
                get_config_file_paths(base_model_path_, base_model_config_path_, "single-speaker-english-fast");
        else if (model_ == "single_speaker_fast")
            config_file_paths = get_config_file_paths(base_model_path_, base_model_config_path_, "single-speaker-fast");

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

            CONFIG_AUTO_SET(file_body["mode_param"], spacker_role);
            CONFIG_AUTO_SET(file_body["mode_param"], spacker_speed);
            CONFIG_AUTO_SET(file_body["mode_param"], ttsModelName);
            mode_config_.ttsModelName = base_model + mode_config_.ttsModelName;
            if (config_body.contains("awake_delay"))
                awake_delay_ = config_body["awake_delay"].get<int>();
            else if (file_body["mode_param"].contains("awake_delay"))
                awake_delay_ = file_body["mode_param"]["awake_delay"];
            int32_t modelSize = ttsLoadModel((char *)mode_config_.ttsModelName.c_str(), &dataW);
            synthesizer_      = std::make_unique<SynthesizerTrn>(dataW, modelSize);
            int32_t spkNum    = synthesizer_->getSpeakerNum();
            SLOGI("Available speakers in the model are %d", spkNum);
        } catch (...) {
            SLOGE("config false");
            return -3;
        }
        return 0;
    }

    bool TTS(const std::string &msg)
    {
        SLOGI("TTS msg:%s", msg.c_str());
        int32_t dataLen;
        int16_t *rawData = synthesizer_->infer(msg, mode_config_.spacker_role, mode_config_.spacker_speed, dataLen);
        if (!rawData) {
            SLOGW("tts infer false!");
            return true;
        }
        std::vector<int16_t> wavData(rawData, rawData + dataLen);
        out_callback_(std::string((char *)rawData, dataLen * sizeof(int16_t)), true);
        free(rawData);
        return false;
    }

    bool delete_model()
    {
        synthesizer_.reset();
        return true;
    }

    llm_task(const std::string &workid)
    {
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
    }
};

#undef CONFIG_AUTO_SET

class llm_tts : public StackFlow {
private:
    int task_count_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_tts() : StackFlow("tts")
    {
        task_count_ = 1;
    }

    void task_output(const std::weak_ptr<llm_task> llm_task_obj_weak,
                     const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &data, bool finish)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::string base64_data;
        int len = encode_base64(data, base64_data);
        if (llm_channel->enstream_) {
            static int count = 0;
            nlohmann::json data_body;
            data_body["index"] = count++;
            if (!finish)
                data_body["delta"] = base64_data;
            else
                data_body["delta"] = std::string("");
            data_body["finish"] = finish;
            if (finish) count = 0;
            llm_channel->send(llm_task_obj->response_format_, data_body, LLM_NO_ERROR);
        } else if (finish) {
            llm_channel->send(llm_task_obj->response_format_, base64_data, LLM_NO_ERROR);
        }
        if (llm_task_obj->response_format_.find("sys") != std::string::npos) {
            unit_call("audio", "queue_play", data);
        }
    }

    std::vector<std::string> splitEachChar(const std::string &text)
    {
        std::vector<std::string> words;
        std::string input(text);
        int len = input.length();
        int i   = 0;

        while (i < len) {
            int next = 1;
            if ((input[i] & 0x80) == 0x00) {
            } else if ((input[i] & 0xE0) == 0xC0) {
                next = 2;
            } else if ((input[i] & 0xF0) == 0xE0) {
                next = 3;
            } else if ((input[i] & 0xF8) == 0xF0) {
                next = 4;
            }
            words.push_back(input.substr(i, next));
            i += next;
        }
        return words;
    }

    void task_user_data(const std::weak_ptr<llm_task> llm_task_obj_weak,
                        const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                        const std::string &data)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        if (data.empty() || (data == "None")) return;
        static std::string faster_stream_buff;
        nlohmann::json error_body;
        const std::string *next_data = &data;
        bool enbase64                = (object.find("base64") == std::string::npos) ? false : true;
        bool enstream                = (object.find("stream") == std::string::npos) ? false : true;
        bool finish_flage            = true;
        int ret;
        std::string tmp_msg1;
        if (enstream) {
            std::string finish = sample_json_str_get((*next_data), "finish");
            tmp_msg1           = sample_json_str_get((*next_data), "delta");
            finish_flage       = (finish == "true") ? true : false;
            next_data          = &tmp_msg1;
        }
        std::string tmp_msg2;
        if (enbase64) {
            ret = decode_base64((*next_data), tmp_msg2);
            if (ret == -1) {
                return;
            }
            next_data = &tmp_msg2;
        }
        std::string user_msg              = sample_unescapeString(*next_data);
        std::vector<std::string> tmp_data = splitEachChar(user_msg);
        for (auto cutf8 : tmp_data) {
            if (cutf8 == "，" || cutf8 == "、" || cutf8 == "," || cutf8 == "。" || cutf8 == "." || cutf8 == "!" ||
                cutf8 == "！" || cutf8 == "?" || cutf8 == "？" || cutf8 == ";" || cutf8 == "；") {
                faster_stream_buff += cutf8;
                ret = llm_task_obj->TTS(faster_stream_buff);
                faster_stream_buff.clear();
                if (ret) {
                    error_body["code"]    = -11;
                    error_body["message"] = "Model run failed.";
                    llm_channel->send("None", "None", error_body, llm_channel->work_id_);
                }
            } else {
                faster_stream_buff += cutf8;
            }
        }
        if (finish_flage) {
            if (!faster_stream_buff.empty()) {
                faster_stream_buff.push_back('.');
                ret = llm_task_obj->TTS(faster_stream_buff);
                faster_stream_buff.clear();
                if (ret) {
                    error_body["code"]    = -11;
                    error_body["message"] = "Model run failed.";
                    llm_channel->send("None", "None", error_body, llm_channel->work_id_);
                }
            }
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
        if (llm_task_obj->superior_flage_) {
            llm_channel->stop_subscriber_work_id(llm_task_obj->superior_id_);
            if (llm_task_obj->response_format_.find("sys") != std::string::npos) {
                unit_call("audio", "queue_play_stop", data);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(llm_task_obj->awake_delay_));
            if (llm_task_obj->response_format_.find("sys") != std::string::npos) {
                unit_call("audio", "play_stop", data);
            }
            llm_channel->subscriber_work_id(
                llm_task_obj->superior_id_,
                std::bind(&llm_tts::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
        }
    }

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        if ((llm_task_channel_.size() - 1) == task_count_) {
            error_body["code"]    = -21;
            error_body["message"] = "task full";
            send("None", "None", error_body, "tts");
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
            send("None", "None", error_body, "tts");
            return -2;
        }
        int ret = llm_task_obj->load_model(config_body);
        if (ret == 0) {
            llm_channel->set_output(llm_task_obj->enoutput_);
            llm_channel->set_stream(llm_task_obj->enstream_);
            SLOGI("llm_task_obj->enoutput_:%d", llm_task_obj->enoutput_);
            SLOGI("llm_task_obj->enstream_:%d", llm_task_obj->enstream_);
            llm_task_obj->set_output(std::bind(&llm_tts::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));
            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("tts") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_tts::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if ((input.find("llm") != std::string::npos) || (input.find("vlm") != std::string::npos)) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_tts::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                    llm_task_obj->superior_id_    = input;
                    llm_task_obj->superior_flage_ = true;
                } else if (input.find("kws") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_tts::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
            send("None", "None", error_body, "tts");
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_tts::link:%s", data.c_str());
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
        if ((data.find("llm") != std::string::npos) || (data.find("vlm") != std::string::npos)) {
            ret = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_tts::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->superior_id_    = data;
            llm_task_obj->superior_flage_ = true;
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("kws") != std::string::npos) {
            ret = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_tts::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
        SLOGI("llm_tts::unlink:%s", data.c_str());
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
                break;
            } else {
                ++it;
            }
        }
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    void taskinfo(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_tts::taskinfo:%s", data.c_str());
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("tts.tasklist", req_body, LLM_NO_ERROR, work_id);
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
            send("tts.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_tts::exit:%s", data.c_str());

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
        llm_task_.erase(work_id_num);
        send("None", "None", LLM_NO_ERROR, work_id);
        return 0;
    }

    ~llm_tts()
    {
        while (1) {
            auto iteam = llm_task_.begin();
            if (iteam == llm_task_.end()) {
                break;
            }
            iteam->second->stop();
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
    llm_tts llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}