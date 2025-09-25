/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"

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

#include <onnxruntime_cxx_api.h>

#include "kaldi-native-fbank/csrc/online-feature.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_kws_new will be exit!");
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
    int chunk_size                 = 32;
    float threshold                = 0.9f;
    int min_continuous_frames      = 5;
    int count_frames               = 0;
    long long last_trigger_time_ms = -1e9;
    long long frame_index_global   = 0;
    int REFRACTORY_TIME_MS         = 2000;
    const int RESAMPLE_RATE        = 16000;
    const int FEAT_DIM             = 80;
    std::vector<float> cache;
    std::unique_ptr<Ort::Session> session;
    knf::FbankOptions opts_;
    std::unique_ptr<knf::OnlineFbank> fbank_;
    Ort::Env env;
    Ort::SessionOptions session_options;

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
    int delay_audio_frame_ = 32;
    buffer_t *pcmdata;
    std::string wake_wav_file_;
    int file_counter = 0;

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

            std::string model_file = base_model + "kws.onnx";

            if (config_body.contains("wake_wav_file"))
                wake_wav_file_ = config_body["wake_wav_file"];
            else if (file_body["mode_param"].contains("wake_wav_file"))
                wake_wav_file_ = file_body["mode_param"]["wake_wav_file"];

            session = std::make_unique<Ort::Session>(env, model_file.c_str(), session_options);
            cache.assign(1 * 32 * 88, 0.0f);

            opts_.frame_opts.samp_freq        = 16000;
            opts_.frame_opts.frame_length_ms  = 25.0;
            opts_.frame_opts.frame_shift_ms   = 10.0;
            opts_.frame_opts.snip_edges       = false;
            opts_.frame_opts.dither           = 0.0;
            opts_.frame_opts.preemph_coeff    = 0.97;
            opts_.frame_opts.remove_dc_offset = true;
            opts_.frame_opts.window_type      = "povey";

            opts_.mel_opts.num_bins  = 80;
            opts_.mel_opts.low_freq  = 20;
            opts_.mel_opts.high_freq = 0;

            opts_.energy_floor = 0.0;
            opts_.use_energy   = false;

            opts_.raw_energy = true;
            // use_log_fbank / use_power 由 knf::OnlineFbank 默认控制，一般一致

            fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
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

    bool detect_wakeup(const std::vector<float> &scores)
    {
        bool triggered = false;
        SLOGE("%d", scores.size());
        for (auto score : scores) {
            printf("%f ", score);
            if (score > threshold) {
                count_frames++;
                if (count_frames >= min_continuous_frames) {
                    long long trigger_time_ms = (frame_index_global - min_continuous_frames + 1) * 10;
                    if (trigger_time_ms - last_trigger_time_ms >= REFRACTORY_TIME_MS) {
                        last_trigger_time_ms = trigger_time_ms;
                        triggered            = true;
                    }
                }
            } else {
                count_frames = 0;
            }
            frame_index_global++;
        }
        SLOGE("\n");
        return triggered;
    }

    std::vector<std::vector<float>> compute_fbank_kaldi(const std::vector<float> &waveform, int sample_rate,
                                                        int num_mel_bins)
    {
        fbank_.reset();
        fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
        fbank_->AcceptWaveform(sample_rate, waveform.data(), waveform.size());
        int num_frames = fbank_->NumFramesReady();

        std::vector<std::vector<float>> features;

        features.reserve(num_frames);
        for (int i = 0; i < num_frames; ++i) {
            const float *frame_data = fbank_->GetFrame(i);
            std::vector<float> frame(frame_data, frame_data + num_mel_bins);
            features.push_back(std::move(frame));
        }

        return features;
    }

    std::vector<float> run_inference(const std::vector<float> &audio_chunk_16k)
    {
        std::vector<std::vector<float>> fbank_feats;
        fbank_feats = compute_fbank_kaldi(audio_chunk_16k, RESAMPLE_RATE, FEAT_DIM);

        if (fbank_feats.empty()) {
            return {};
        }

        int T = fbank_feats.size();
        std::vector<float> mat_flattened;
        for (const auto &feat : fbank_feats) {
            mat_flattened.insert(mat_flattened.end(), feat.begin(), feat.end());
        }

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(T), FEAT_DIM};
        std::vector<int64_t> cache_shape = {1, 32, 88};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor     = Ort::Value::CreateTensor<float>(
            memory_info, mat_flattened.data(), mat_flattened.size(), input_shape.data(), input_shape.size());
        Ort::Value cache_tensor = Ort::Value::CreateTensor<float>(memory_info, cache.data(), cache.size(),
                                                                  cache_shape.data(), cache_shape.size());

        const char *input_names[]  = {"input", "cache"};
        const char *output_names[] = {"output", "r_cache"};

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        inputs.push_back(std::move(cache_tensor));

        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 2, output_names, 2);

        float *out_data       = output_tensors[0].GetTensorMutableData<float>();
        float *cache_out_data = output_tensors[1].GetTensorMutableData<float>();

        std::vector<int64_t> out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t out_size                = 1;
        for (auto dim : out_shape) out_size *= dim;

        std::vector<float> out_chunk(out_data, out_data + out_size);
        std::copy(cache_out_data, cache_out_data + cache.size(), cache.begin());

        return out_chunk;
    }

    void sys_pcm_on_data(const std::string &raw)
    {
        static int count = 0;
        if (count < delay_audio_frame_) {
            buffer_write_char(pcmdata, raw.data(), raw.length());
            count++;
            return;
        }
        buffer_write_char(pcmdata, raw.data(), raw.length());
        buffer_position_set(pcmdata, 0);

        std::vector<float> floatSamples;
        std::vector<int16_t> int16Samples;
        {
            int16_t audio_val;
            while (buffer_read_i16(pcmdata, &audio_val, 1)) {
                int16Samples.push_back(audio_val);
                float normalizedSample = static_cast<float>(audio_val) / 1.0f;
                float sample           = static_cast<float>(normalizedSample);
                floatSamples.push_back(sample);
            }
        }

        buffer_resize(pcmdata, 0);
        count = 0;

        auto scores = run_inference(floatSamples);
        if (detect_wakeup(scores)) {
            if (enwake_audio_ && (!wake_wav_file_.empty()) && play_awake_wav) {
                play_awake_wav(wake_wav_file_);
            }
            if (out_callback_) out_callback_("", true);
        }
    }

    void trigger()
    {
        if (out_callback_) out_callback_("", true);
    }

    bool delete_model()
    {
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
    llm_kws() : StackFlow("kws_new")
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
                    llm_task_obj->delay_audio_frame_ = 0;
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