/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/parse-options.h"

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <stdexcept>
#include <thread_safe_list.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"

#define BUFFER_IMPLEMENTATION
#include <stdbool.h>
#include <stdint.h>
#include "libs/buffer.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_kws will be exit!");
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
    sherpa_onnx::KeywordSpotterConfig mode_config_;
    std::unique_ptr<sherpa_onnx::KeywordSpotter> spotter_;
    std::unique_ptr<sherpa_onnx::OnlineStream> spotter_stream_;

public:
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    std::vector<std::string> kws_;
    bool enoutput_;
    bool enoutput_json_;
    bool enstream_;
    bool enwake_audio_;
    std::atomic_bool audio_flage_;
    task_callback_t out_callback_;
    int delay_audio_frame_ = 100;
    buffer_t *pcmdata;
    std::string wake_wav_file_;

    std::function<void(const std::string &)> play_awake_wav;

    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            model_           = config_body.at("model");
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");
            if (config_body.contains("enwake_audio")) {
                enwake_audio_ = config_body["enwake_audio"];
            } else {
                enwake_audio_ = true;
            }
            if (config_body.contains("input")) {
                if (config_body["input"].is_string()) {
                    inputs_.push_back(config_body["input"].get<std::string>());
                } else if (config_body["input"].is_array()) {
                    for (auto _in : config_body["input"]) {
                        inputs_.push_back(_in.get<std::string>());
                    }
                }
            }
            if (config_body.contains("kws")) {
                if (config_body["kws"].is_string()) {
                    kws_.push_back(config_body["kws"].get<std::string>());
                } else if (config_body["kws"].is_array()) {
                    for (auto _in : config_body["kws"]) {
                        kws_.push_back(_in.get<std::string>());
                    }
                }
            }
            enoutput_json_ = response_format_.find("json") == std::string::npos ? false : true;
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
            CONFIG_AUTO_SET(file_body["mode_param"], feat_config.low_freq);
            CONFIG_AUTO_SET(file_body["mode_param"], feat_config.high_freq);
            CONFIG_AUTO_SET(file_body["mode_param"], feat_config.dither);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.encoder);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.decoder);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.joiner);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.paraformer.encoder);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.paraformer.decoder);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.wenet_ctc.model);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.wenet_ctc.chunk_size);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.wenet_ctc.num_left_chunks);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer2_ctc.model);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.nemo_ctc.model);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.device);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.provider);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.cuda_config.cudnn_conv_algo_search);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_max_workspace_size);
            CONFIG_AUTO_SET(file_body["mode_param"],
                            model_config.provider_config.trt_config.trt_max_partition_iterations);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_min_subgraph_size);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_fp16_enable);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_detailed_build_log);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_engine_cache_enable);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_engine_cache_path);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_timing_cache_enable);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_timing_cache_path);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.trt_config.trt_dump_subgraphs);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.tokens);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.num_threads);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.warm_up);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.debug);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.model_type);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.modeling_unit);
            CONFIG_AUTO_SET(file_body["mode_param"], model_config.bpe_vocab);
            CONFIG_AUTO_SET(file_body["mode_param"], max_active_paths);
            CONFIG_AUTO_SET(file_body["mode_param"], num_trailing_blanks);
            CONFIG_AUTO_SET(file_body["mode_param"], keywords_score);
            CONFIG_AUTO_SET(file_body["mode_param"], keywords_threshold);
            CONFIG_AUTO_SET(file_body["mode_param"], keywords_file);

            if (config_body.contains("wake_wav_file"))
                wake_wav_file_ = config_body["wake_wav_file"];
            else if (file_body["mode_param"].contains("wake_wav_file"))
                wake_wav_file_ = file_body["mode_param"]["wake_wav_file"];

            mode_config_.model_config.transducer.encoder = base_model + mode_config_.model_config.transducer.encoder;
            mode_config_.model_config.transducer.decoder = base_model + mode_config_.model_config.transducer.decoder;
            mode_config_.model_config.transducer.joiner  = base_model + mode_config_.model_config.transducer.joiner;
            mode_config_.model_config.tokens             = base_model + mode_config_.model_config.tokens;
            mode_config_.keywords_file                   = base_model + mode_config_.keywords_file;

            std::ofstream temp_awake_key("/tmp/kws_awake.txt.tmp");
            for (const auto &keyword : kws_) {
                temp_awake_key << keyword << std::endl;
            }
            temp_awake_key.close();
            std::ostringstream awake_key_compile_cmd;
            if (file_exists("/opt/m5stack/scripts/text2token.py"))
                awake_key_compile_cmd << "PYTHONPATH=/opt/m5stack/lib/sherpa-onnx/site-packages /usr/bin/python3 "
                                         "/opt/m5stack/scripts/text2token.py ";
            else if (file_exists("/opt/m5stack/scripts/llm-kws_text2token.py"))
                awake_key_compile_cmd << "PYTHONPATH=/opt/m5stack/lib/sherpa-onnx/site-packages /usr/bin/python3 "
                                         "/opt/m5stack/scripts/llm-kws_text2token.py ";
            else {
                SLOGE("text2token.py or llm-kws_text2token.py not found!");
            }
            awake_key_compile_cmd << "--text /tmp/kws_awake.txt.tmp ";
            awake_key_compile_cmd << "--tokens " << mode_config_.model_config.tokens << " ";
            if (file_body["mode_param"].contains("text2token-tokens-type")) {
                awake_key_compile_cmd << "--tokens-type "
                                      << file_body["mode_param"]["text2token-tokens-type"].get<std::string>() << " ";
            }
            if (file_body["mode_param"].contains("text2token-bpe-model")) {
                awake_key_compile_cmd << "--bpe-model " << base_model
                                      << file_body["mode_param"]["text2token-bpe-model"].get<std::string>() << " ";
            }
            awake_key_compile_cmd << "--output " << mode_config_.keywords_file;
            system(awake_key_compile_cmd.str().c_str());
            spotter_        = std::make_unique<sherpa_onnx::KeywordSpotter>(mode_config_);
            spotter_stream_ = spotter_->CreateStream();
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
        spotter_stream_->AcceptWaveform(mode_config_.feat_config.sampling_rate, floatSamples.data(),
                                        floatSamples.size());
        while (spotter_->IsReady(spotter_stream_.get())) {
            spotter_->DecodeStream(spotter_stream_.get());
        }
        sherpa_onnx::KeywordResult r = spotter_->GetResult(spotter_stream_.get());
        if (!r.keyword.empty()) {
            if (enwake_audio_ && (!wake_wav_file_.empty()) && play_awake_wav) {
                play_awake_wav(wake_wav_file_);
            }
            if (out_callback_) {
                if (enoutput_json_)
                    out_callback_(r.AsJsonString(), true);
                else
                    out_callback_("", true);
            }
        }
    }

    void trigger()
    {
        if (out_callback_) out_callback_("", true);
    }

    bool delete_model()
    {
        spotter_.reset();
        return true;
    }

    llm_task(const std::string &workid) : audio_flage_(false)
    {
        pcmdata = buffer_create();
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

class llm_kws : public StackFlow {
private:
    enum { EVENT_TRIGGER = EVENT_EXPORT + 1 };
    int task_count_;
    std::string audio_url_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_kws() : StackFlow("kws")
    {
        task_count_ = 1;
        event_queue_.appendListener(EVENT_TRIGGER, std::bind(&llm_kws::trigger, this, std::placeholders::_1));
        rpc_ctx_->register_rpc_action(
            "trigger", [this](pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) -> std::string {
                this->event_queue_.enqueue(EVENT_TRIGGER,
                                           std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
                return LLM_NONE;
            });
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
        if (data.empty()) {
            llm_channel->send(llm_task_obj->response_format_, true, LLM_NO_ERROR);
            return;
        }
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

    void play_awake_wav(const std::string &wav_file)
    {
        FILE *fp = fopen(wav_file.c_str(), "rb");
        if (!fp) {
            printf("Open %s failed!\n", wav_file.c_str());
            return;
        }
        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        std::vector<char> wav_data(size);
        fread(wav_data.data(), 1, wav_data.size(), fp);
        fclose(fp);
        int post = 0;
        for (int i = 0; i < wav_data.size() - 4; i++) {
            if ((wav_data[i] == 'd') && (wav_data[i + 1] == 'a') && (wav_data[i + 2] == 't') &&
                (wav_data[i + 3] == 'a')) {
                post = i + 8;
                break;
            }
        }
        if (post != 0) {
            unit_call("audio", "play_raw", std::string((char *)(wav_data.data() + post), size - post));
        }
    }

    void task_pause(const std::weak_ptr<llm_task> llm_task_obj_weak,
                    const std::weak_ptr<llm_channel_obj> llm_channel_weak)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        if (llm_task_obj->audio_flage_) {
            if (!audio_url_.empty()) llm_channel->stop_subscriber(audio_url_);
            llm_task_obj->audio_flage_ = false;
        }
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
        task_pause(llm_task_[work_id_num], get_channel(work_id_num));
        send("None", "None", LLM_NO_ERROR, work_id);
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

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        if ((llm_task_channel_.size() - 1) == task_count_) {
            error_body["code"]    = -21;
            error_body["message"] = "task full";
            send("None", "None", error_body, "kws");
            return -1;
        }

        int work_id_num                             = sample_get_work_id_num(work_id);
        auto llm_channel                            = get_channel(work_id);
        auto llm_task_obj                           = std::make_shared<llm_task>(work_id);
        std::weak_ptr<llm_task> _llm_task_obj       = llm_task_obj;
        std::weak_ptr<llm_channel_obj> _llm_channel = llm_channel;
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
            llm_task_obj->play_awake_wav = std::bind(&llm_kws::play_awake_wav, this, std::placeholders::_1);
            llm_task_obj->set_output(std::bind(&llm_kws::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("sys") != std::string::npos) {
                    audio_url_ = unit_call("audio", "cap", "None");
                    llm_channel->subscriber(audio_url_,
                                            [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                                                auto llm_task_obj = _llm_task_obj.lock();
                                                if (llm_task_obj) llm_task_obj->sys_pcm_on_data(raw->string());
                                            });
                    llm_task_obj->audio_flage_ = true;
                } else if (input.find("kws") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_kws::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
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
            send("None", "None", error_body, "kws");
            return -1;
        }
    }

    void taskinfo(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_kws::taskinfo:%s", data.c_str());
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("kws.tasklist", req_body, LLM_NO_ERROR, work_id);
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
            send("kws.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_kws::exit:%s", data.c_str());
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

    std::string trigger(const std::shared_ptr<void> &arg)
    {
        std::shared_ptr<stackflow_data> originalPtr = std::static_pointer_cast<stackflow_data>(arg);
        std::string zmq_url                         = originalPtr->string(0);
        std::string data                            = originalPtr->string(1);
        std::string work_id                         = sample_json_str_get(data, "work_id");
        if (work_id.length() == 0) {
            nlohmann::json out_body;
            out_body["request_id"]       = sample_json_str_get(data, "request_id");
            out_body["work_id"]          = "kws";
            out_body["created"]          = time(NULL);
            out_body["object"]           = "";
            out_body["data"]             = "";
            out_body["error"]["code"]    = -2;
            out_body["error"]["message"] = "json format error.";
            pzmq _zmq(zmq_url, ZMQ_PUSH);
            std::string out = out_body.dump();
            out += "\n";
            _zmq.send_data(out);
            return LLM_NONE;
        }

        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            nlohmann::json out_body;
            out_body["request_id"]       = sample_json_str_get(data, "request_id");
            out_body["work_id"]          = "kws";
            out_body["created"]          = time(NULL);
            out_body["object"]           = "";
            out_body["data"]             = "";
            out_body["error"]["code"]    = -6;
            out_body["error"]["message"] = "Unit Does Not Exist";
            pzmq _zmq(zmq_url, ZMQ_PUSH);
            std::string out = out_body.dump();
            out += "\n";
            _zmq.send_data(out);
            return LLM_NONE;
        }
        llm_task_[work_id_num]->trigger();
        return LLM_NONE;
    }

    ~llm_kws()
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
    llm_kws llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    return 0;
}