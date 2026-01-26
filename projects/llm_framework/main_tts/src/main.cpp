/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "utils.h"
#include "SynthesizerTrn.h"
#include "Kokoro.h"
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <stdexcept>
#include <samplerate.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_tts will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;

typedef struct {
    std::string axmodel_dir;
    std::string lang;
    std::string voice_path;
    std::string voice_name;
    std::string vocab_path;
    std::string espeak_data_path;
    std::string dict_dir;
    float spacker_speed = 1.0;
    int max_len         = 96;
    int mode_rate       = 44100;
    int audio_rate      = 16000;
    float speed         = 1.0f;
    float pause         = 0.0f;
    float fade_out      = 0.0f;

} kokoro_config;

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

#define CONFIG_KOKORO_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))                  \
        mode_kokoro_config_.key = config_body[#key]; \
    else if (obj.contains(#key))                     \
        mode_kokoro_config_.key = obj[#key];

class llm_task {
private:
    float *dataW = NULL;
    int modelSize;

public:
    kokoro_config mode_kokoro_config_;
    std::unique_ptr<SynthesizerTrn> synthesizer_;
    std::unique_ptr<kokoro::Kokoro> kokoro_;
    std::string model_type_;
    std::string model_;
    SynthesizerTrn_config mode_config_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    bool enoutput_;
    bool enstream_;
    std::atomic_bool superior_flage_;
    std::string superior_id_;
    static int ax_init_flage_;
    task_callback_t out_callback_;
    bool enaudio_;
    int awake_delay_ = 1000;

    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            model_           = config_body.at("model");
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");

            if (model_.rfind("single-speaker", 0) == 0) {
                model_type_ = "summer_tts";
            } else {
                model_type_ = "kokoro_tts";
            }

            if (config_body.contains("enaudio")) enaudio_ = config_body.at("enaudio");
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

    int load_summer_tts_model(const nlohmann::json &config_body)
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
            return -6;
        }
        return 0;
    }

    int load_kokoro_tts_model(const nlohmann::json &config_body)
    {
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
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], axmodel_dir);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], lang);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], voice_path);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], voice_name);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], vocab_path);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], espeak_data_path);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], dict_dir);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], spacker_speed);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], max_len);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], mode_rate);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], audio_rate);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], speed);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], pause);
            CONFIG_KOKORO_AUTO_SET(file_body["mode_param"], fade_out);
            mode_kokoro_config_.axmodel_dir      = base_model + mode_kokoro_config_.axmodel_dir;
            mode_kokoro_config_.voice_path       = base_model + mode_kokoro_config_.voice_path;
            mode_kokoro_config_.vocab_path       = base_model + mode_kokoro_config_.vocab_path;
            mode_kokoro_config_.espeak_data_path = base_model + mode_kokoro_config_.espeak_data_path;
            mode_kokoro_config_.dict_dir         = base_model + mode_kokoro_config_.dict_dir;
            kokoro_                              = std::make_unique<kokoro::Kokoro>();
            if (!kokoro_->init(mode_kokoro_config_.axmodel_dir, mode_kokoro_config_.max_len, mode_kokoro_config_.lang,
                               mode_kokoro_config_.voice_path, mode_kokoro_config_.voice_name,
                               mode_kokoro_config_.vocab_path, mode_kokoro_config_.espeak_data_path,
                               mode_kokoro_config_.dict_dir)) {
                SLOGE("encoder init failed!");
                return -4;
            }
        } catch (...) {
            SLOGE("config false");
            return -6;
        }
        return 0;
    }

    int load_model(const nlohmann::json &config_body)
    {
        if (parse_config(config_body)) {
            return -1;
        }
        if (model_type_ == "summer_tts") {
            SLOGI("load summer tts model");
            return load_summer_tts_model(config_body);
        } else {
            SLOGI("load kokoro tts model");
            return load_kokoro_tts_model(config_body);
        }
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

    void resample_audio(const float *input_buffer, int input_length, float *output_buffer, int *output_length,
                        double src_ratio)
    {
        int error            = 0;
        SRC_STATE *src_state = src_new(SRC_SINC_FASTEST, 1, &error);
        if (!src_state) {
            SLOGE("src_new failed: %s", src_strerror(error));
            *output_length = 0;
            return;
        }

        SRC_DATA src_data;
        memset(&src_data, 0, sizeof(src_data));
        src_data.data_in       = input_buffer;
        src_data.input_frames  = input_length;
        src_data.data_out      = output_buffer;
        src_data.output_frames = static_cast<long>(input_length * src_ratio) + 1;
        src_data.src_ratio     = src_ratio;
        src_data.end_of_input  = 1;

        error = src_process(src_state, &src_data);
        if (error) {
            SLOGE("src_process failed: %s", src_strerror(error));
            *output_length = 0;
            src_delete(src_state);
            return;
        }

        *output_length = src_data.output_frames_gen;
        src_delete(src_state);
    }

    bool TTS(const std::string &msg, bool finish)
    {
        SLOGI("TTS msg:%s", msg.c_str());
        if (msg.empty()) {
            if (out_callback_) {
                out_callback_(std::string(), finish);
            }
            return false;
        }

        if (model_type_ == "summer_tts") {
            int32_t dataLen  = 0;
            int16_t *rawData = synthesizer_->infer(msg, mode_config_.spacker_role, mode_config_.spacker_speed, dataLen);

            if (!rawData || dataLen <= 0) {
                SLOGW("summer tts infer failed!");
                return true;
            }

            if (out_callback_) {
                out_callback_(std::string(reinterpret_cast<char *>(rawData), dataLen * sizeof(int16_t)), finish);
            }

            free(rawData);
            return false;

        } else if (model_type_ == "kokoro_tts") {
            std::vector<float> audio;
            int src_rate = mode_kokoro_config_.mode_rate;
            int dst_rate = mode_kokoro_config_.audio_rate;

            if (!kokoro_->tts(msg, mode_kokoro_config_.voice_name, mode_kokoro_config_.speed, src_rate,
                              mode_kokoro_config_.fade_out, mode_kokoro_config_.pause, audio)) {
                SLOGE("kokoro tts run failed!");
                return true;
            }

            if (audio.empty()) {
                if (out_callback_) {
                    out_callback_(std::string(), finish);
                }
                SLOGE("kokoro tts audio empty!");
                return false;
            }

            std::vector<float> resampled_pcm;
            const float *pcm_ptr = audio.data();
            int pcm_len          = static_cast<int>(audio.size());

            if (src_rate != dst_rate) {
                double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);

                int max_dst_len = static_cast<int>(pcm_len * ratio) + 1;
                resampled_pcm.resize(max_dst_len);

                int out_len = 0;
                resample_audio(pcm_ptr, pcm_len, resampled_pcm.data(), &out_len, ratio);
                resampled_pcm.resize(out_len);
                pcm_ptr = resampled_pcm.data();
                pcm_len = out_len;
            }

            std::vector<int16_t> wav_pcm_data;
            wav_pcm_data.reserve(pcm_len);
            for (int i = 0; i < pcm_len; ++i) {
                float v = pcm_ptr[i];

                if (std::abs(v) > 0.95f) {
                    v = v > 0 ? 0.95f : -0.95f;
                }

                wav_pcm_data.push_back(static_cast<int16_t>(v * INT16_MAX));
            }

            if (out_callback_) {
                out_callback_(
                    std::string(reinterpret_cast<char *>(wav_pcm_data.data()), wav_pcm_data.size() * sizeof(int16_t)),
                    finish);
            }

            return false;
        }

        return true;
    }

    bool delete_model()
    {
        synthesizer_.reset();
        return true;
    }

    void _ax_init()
    {
        if (!ax_init_flage_) {
            int ret = AX_SYS_Init();
            if (0 != ret) {
                fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
            }
            AX_ENGINE_NPU_ATTR_T npu_attr;
            memset(&npu_attr, 0, sizeof(npu_attr));
            ret = AX_ENGINE_Init(&npu_attr);
            if (0 != ret) {
                fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
            }
        }
        ax_init_flage_++;
    }

    void _ax_deinit()
    {
        if (ax_init_flage_ > 0) {
            --ax_init_flage_;
            if (!ax_init_flage_) {
                AX_ENGINE_Deinit();
                AX_SYS_Deinit();
            }
        }
    }

    llm_task(const std::string &workid)
    {
        enaudio_ = true;
        _ax_init();
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
        if (kokoro_) kokoro_.reset();
        _ax_deinit();
    }
};
int llm_task::ax_init_flage_ = 0;
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
        if (!data.empty()) {
            int len = encode_base64(data, base64_data);
        }
        if (llm_channel->enstream_) {
            static int count = 0;
            nlohmann::json data_body;
            data_body["index"] = count++;
            if (!data.empty())
                data_body["delta"] = base64_data;
            else
                data_body["delta"] = "";
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
            std::string finish_str = sample_json_str_get((*next_data), "finish");
            finish_flage           = (finish_str.find("true") != std::string::npos);
            tmp_msg1               = sample_json_str_get((*next_data), "delta");
            next_data              = &tmp_msg1;
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
                ret = llm_task_obj->TTS(faster_stream_buff, false);
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
                ret = llm_task_obj->TTS(faster_stream_buff, true);
                faster_stream_buff.clear();
                if (ret) {
                    error_body["code"]    = -11;
                    error_body["message"] = "Model run failed.";
                    llm_channel->send("None", "None", error_body, llm_channel->work_id_);
                }
            }
            llm_task_obj->TTS("", true);
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