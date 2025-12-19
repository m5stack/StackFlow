/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"

#include "EngineWrapper.hpp"
#include "ax_sys_api.h"

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

#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

typedef struct mode_config_axera {
    int chunk_size            = 32;
    float threshold           = 0.9f;
    int min_continuous_frames = 5;
    int REFRACTORY_TIME_MS    = 2000;
    int RESAMPLE_RATE         = 16000;
    int FEAT_DIM              = 80;
    int delay_audio_frame_    = 32;
} kws_config_axera;

class llm_task {
private:
    std::string model_type_;
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    std::vector<std::string> kws_;
    bool enoutput_      = true;
    bool enoutput_json_ = false;
    bool enstream_      = false;
    bool enwake_audio_  = true;
    std::atomic_bool audio_flage_;
    static int ax_init_flage_;
    task_callback_t out_callback_;
    buffer_t *pcmdata;
    std::string wake_wav_file_;
    std::function<void(const std::string &)> play_awake_wav;
    int delay_audio_frame_ = 10;

    sherpa_onnx::KeywordSpotterConfig sherpa_config_;
    std::unique_ptr<sherpa_onnx::KeywordSpotter> sherpa_spotter_;
    std::unique_ptr<sherpa_onnx::OnlineStream> sherpa_stream_;

    kws_config_axera axera_config_;
    std::vector<float> axera_cache_;
    std::unique_ptr<EngineWrapper> axera_session_;
    knf::FbankOptions fbank_opts_;
    std::unique_ptr<knf::OnlineFbank> fbank_;
    Ort::SessionOptions session_options_;
    int count_frames_               = 0;
    long long last_trigger_time_ms_ = -1e9;
    long long frame_index_global_   = 0;
    int last_btn_204_state          = -1;

public:
    inline const std::string &model() const
    {
        return model_;
    }
    inline const std::string &response_format() const
    {
        return response_format_;
    }
    inline const std::vector<std::string> &inputs() const
    {
        return inputs_;
    }
    inline bool enoutput() const
    {
        return enoutput_;
    }
    bool enstream_flag() const
    {
        return enstream_;
    }

    friend class llm_kws;

    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            model_           = config_body.at("model");
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");

            if (model_.rfind("sherpa-onnx", 0) == 0) {
                model_type_ = "sherpa";
            } else {
                model_type_ = "axera";
            }

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
            enoutput_json_ = response_format_.find("json") != std::string::npos;
            enstream_      = response_format_.find("stream") != std::string::npos;
        } catch (...) {
            SLOGE("setup config_body error");
            return true;
        }
        return false;
    }

#define CONFIG_AUTO_SET_SHERPA(obj, key)        \
    if (config_body.contains(#key))             \
        sherpa_config_.key = config_body[#key]; \
    else if (obj.contains(#key))                \
        sherpa_config_.key = obj[#key];

    int load_model_sherpa(const nlohmann::json &config_body)
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

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], feat_config.sampling_rate);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], feat_config.feature_dim);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], feat_config.low_freq);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], feat_config.high_freq);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], feat_config.dither);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.transducer.encoder);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.transducer.decoder);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.transducer.joiner);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.paraformer.encoder);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.paraformer.decoder);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.wenet_ctc.model);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.wenet_ctc.chunk_size);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.wenet_ctc.num_left_chunks);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.zipformer2_ctc.model);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.nemo_ctc.model);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.provider_config.device);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.provider_config.provider);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.cuda_config.cudnn_conv_algo_search);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_max_workspace_size);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_max_partition_iterations);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_min_subgraph_size);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.provider_config.trt_config.trt_fp16_enable);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_detailed_build_log);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_engine_cache_enable);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_engine_cache_path);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_timing_cache_enable);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"],
                                   model_config.provider_config.trt_config.trt_timing_cache_path);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.provider_config.trt_config.trt_dump_subgraphs);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.tokens);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.num_threads);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.warm_up);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.debug);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.model_type);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.modeling_unit);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], model_config.bpe_vocab);

            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], max_active_paths);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], num_trailing_blanks);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], keywords_score);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], keywords_threshold);
            CONFIG_AUTO_SET_SHERPA(file_body["mode_param"], keywords_file);

            if (config_body.contains("wake_wav_file"))
                wake_wav_file_ = config_body["wake_wav_file"];
            else if (file_body["mode_param"].contains("wake_wav_file"))
                wake_wav_file_ = file_body["mode_param"]["wake_wav_file"];

            sherpa_config_.model_config.transducer.encoder =
                base_model + sherpa_config_.model_config.transducer.encoder;
            sherpa_config_.model_config.transducer.decoder =
                base_model + sherpa_config_.model_config.transducer.decoder;
            sherpa_config_.model_config.transducer.joiner = base_model + sherpa_config_.model_config.transducer.joiner;
            sherpa_config_.model_config.tokens            = base_model + sherpa_config_.model_config.tokens;
            sherpa_config_.keywords_file                  = base_model + sherpa_config_.keywords_file;

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
            awake_key_compile_cmd << "--tokens " << sherpa_config_.model_config.tokens << " ";
            if (file_body["mode_param"].contains("text2token-tokens-type")) {
                awake_key_compile_cmd << "--tokens-type "
                                      << file_body["mode_param"]["text2token-tokens-type"].get<std::string>() << " ";
            }
            if (file_body["mode_param"].contains("text2token-bpe-model")) {
                awake_key_compile_cmd << "--bpe-model " << base_model
                                      << file_body["mode_param"]["text2token-bpe-model"].get<std::string>() << " ";
            }
            awake_key_compile_cmd << "--output " << sherpa_config_.keywords_file;
            system(awake_key_compile_cmd.str().c_str());

            sherpa_spotter_ = std::make_unique<sherpa_onnx::KeywordSpotter>(sherpa_config_);
            sherpa_stream_  = sherpa_spotter_->CreateStream();
        } catch (...) {
            SLOGE("config file read false");
            return -3;
        }

        delay_audio_frame_ = 10;
        return 0;
    }
#undef CONFIG_AUTO_SET_SHERPA

#define CONFIG_AUTO_SET_AXERA(obj, key)        \
    if (config_body.contains(#key))            \
        axera_config_.key = config_body[#key]; \
    else if (obj.contains(#key))               \
        axera_config_.key = obj[#key];

#define OPTS_AUTO_SET(obj, key)              \
    if (config_body.contains(#key))          \
        fbank_opts_.key = config_body[#key]; \
    else if (obj.contains(#key))             \
        fbank_opts_.key = obj[#key];

    int load_model_axera(const nlohmann::json &config_body)
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
            std::string model_file = base_model + "kws.axmodel";

            if (config_body.contains("wake_wav_file"))
                wake_wav_file_ = config_body["wake_wav_file"];
            else if (file_body["mode_param"].contains("wake_wav_file"))
                wake_wav_file_ = file_body["mode_param"]["wake_wav_file"];

            axera_session_ = std::make_unique<EngineWrapper>();
            if (0 != axera_session_->Init(model_file.c_str())) {
                SLOGE("Init axera model failed!");
                return -5;
            }

            axera_cache_.assign(1 * 32 * 88, 0.0f);

            auto &mp = file_body["mode_param"];
            CONFIG_AUTO_SET_AXERA(mp, chunk_size);
            CONFIG_AUTO_SET_AXERA(mp, threshold);
            CONFIG_AUTO_SET_AXERA(mp, min_continuous_frames);
            CONFIG_AUTO_SET_AXERA(mp, REFRACTORY_TIME_MS);
            CONFIG_AUTO_SET_AXERA(mp, RESAMPLE_RATE);
            CONFIG_AUTO_SET_AXERA(mp, FEAT_DIM);
            CONFIG_AUTO_SET_AXERA(mp, delay_audio_frame_);

            OPTS_AUTO_SET(mp, frame_opts.samp_freq);
            OPTS_AUTO_SET(mp, frame_opts.frame_length_ms);
            OPTS_AUTO_SET(mp, frame_opts.frame_shift_ms);
            OPTS_AUTO_SET(mp, frame_opts.snip_edges);
            OPTS_AUTO_SET(mp, frame_opts.dither);
            OPTS_AUTO_SET(mp, frame_opts.preemph_coeff);
            OPTS_AUTO_SET(mp, frame_opts.remove_dc_offset);
            OPTS_AUTO_SET(mp, frame_opts.window_type);
            OPTS_AUTO_SET(mp, mel_opts.num_bins);
            OPTS_AUTO_SET(mp, mel_opts.low_freq);
            OPTS_AUTO_SET(mp, mel_opts.high_freq);
            OPTS_AUTO_SET(mp, energy_floor);
            OPTS_AUTO_SET(mp, use_energy);
            OPTS_AUTO_SET(mp, raw_energy);

            fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
        } catch (...) {
            SLOGE("config file read false");
            return -3;
        }
        delay_audio_frame_ = axera_config_.delay_audio_frame_;
        return 0;
    }
#undef CONFIG_AUTO_SET_ONNX
#undef OPTS_AUTO_SET

    bool detect_wakeup(const std::vector<float> &scores)
    {
        bool triggered = false;
        for (auto score : scores) {
            if (score > axera_config_.threshold) {
                count_frames_++;
                if (count_frames_ >= axera_config_.min_continuous_frames) {
                    long long trigger_time_ms = (frame_index_global_ - axera_config_.min_continuous_frames + 1) * 10;
                    if (trigger_time_ms - last_trigger_time_ms_ >= axera_config_.REFRACTORY_TIME_MS) {
                        last_trigger_time_ms_ = trigger_time_ms;
                        triggered             = true;
                    }
                }
            } else {
                count_frames_ = 0;
            }
            frame_index_global_++;
        }
        return triggered;
    }

    std::vector<std::vector<float>> compute_fbank_kaldi(const std::vector<float> &waveform, int sample_rate,
                                                        int num_mel_bins)
    {
        fbank_.reset();
        fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
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
        std::vector<std::vector<float>> fbank_feats =
            compute_fbank_kaldi(audio_chunk_16k, axera_config_.RESAMPLE_RATE, axera_config_.FEAT_DIM);
        if (fbank_feats.empty()) return {};

        constexpr int FIX_T = 32;
        const int FEAT_DIM  = axera_config_.FEAT_DIM;

        std::vector<float> mat_flattened;
        mat_flattened.resize(FIX_T * FEAT_DIM, 0.0f);

        const int T_in   = static_cast<int>(fbank_feats.size());
        const int T_copy = std::min(T_in, FIX_T);

        for (int t = 0; t < T_copy; ++t) {
            if ((int)fbank_feats[t].size() < FEAT_DIM) continue;
            std::memcpy(mat_flattened.data() + t * FEAT_DIM, fbank_feats[t].data(), sizeof(float) * FEAT_DIM);
        }

        axera_session_->SetInput(mat_flattened.data(), 0);

        axera_session_->SetInput(axera_cache_.data(), 1);

        int ret = axera_session_->Run();
        if (ret) {
            SLOGE("axera_session run failed!");
            return {};
        }

        const float *out_ptr = reinterpret_cast<const float *>(axera_session_->GetOutputPtr(0));
        size_t out_size_f    = axera_session_->GetOutputSize(0) / sizeof(float);
        std::vector<float> out_chunk(out_ptr, out_ptr + out_size_f);

        const float *cache_ptr = reinterpret_cast<const float *>(axera_session_->GetOutputPtr(1));
        size_t cache_size_f    = axera_session_->GetOutputSize(1) / sizeof(float);
        if (cache_size_f != axera_cache_.size()) {
            SLOGE("cache size mismatch: out=%zu, local=%zu", cache_size_f, axera_cache_.size());
            return out_chunk;
        }
        std::memcpy(axera_cache_.data(), cache_ptr, axera_cache_.size() * sizeof(float));

        return out_chunk;
    }

    int load_model(const nlohmann::json &config_body)
    {
        if (parse_config(config_body)) {
            return -1;
        }
        if (model_type_ == "axera") {
            SLOGI("load axera kws model");
            return load_model_axera(config_body);
        } else {
            SLOGI("load sherpa kws model");
            return load_model_sherpa(config_body);
        }
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
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

        int16_t audio_val;
        while (buffer_read_i16(pcmdata, &audio_val, 1)) {
            int16Samples.push_back(audio_val);
            if (model_type_ == "axera") {
                floatSamples.push_back(static_cast<float>(audio_val) / 1.0f);
            } else {
                floatSamples.push_back(static_cast<float>(audio_val) / INT16_MAX);
            }
        }
        buffer_resize(pcmdata, 0);
        count = 0;

        if (model_type_ == "axera") {
            auto scores = run_inference(floatSamples);
            if (detect_wakeup(scores)) {
                if (enwake_audio_ && (!wake_wav_file_.empty()) && play_awake_wav) {
                    play_awake_wav(wake_wav_file_);
                }
                if (out_callback_) {
                    out_callback_("", true);
                }
            }
        } else {
            sherpa_stream_->AcceptWaveform(sherpa_config_.feat_config.sampling_rate, floatSamples.data(),
                                           floatSamples.size());
            while (sherpa_spotter_->IsReady(sherpa_stream_.get())) {
                sherpa_spotter_->DecodeStream(sherpa_stream_.get());
            }
            sherpa_onnx::KeywordResult r = sherpa_spotter_->GetResult(sherpa_stream_.get());
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
    }

    void trigger_wakeup()
    {
        if (enwake_audio_ && (!wake_wav_file_.empty()) && play_awake_wav) {
            play_awake_wav(wake_wav_file_);
        }
        if (out_callback_) {
            if (enoutput_json_)
                out_callback_("{\"reason\":\"button_204\"}", true);
            else
                out_callback_("", true);
        }
    }

    void set_btn_204_state(int state)
    {
        last_btn_204_state = state;
    }

    int get_btn_204_state()
    {
        return last_btn_204_state;
    }

    bool delete_model()
    {
        if (sherpa_spotter_) sherpa_spotter_.reset();
        if (sherpa_stream_) sherpa_stream_.reset();
        if (axera_session_) axera_session_->Release();
        if (fbank_) fbank_.reset();
        return true;
    }

    llm_task(const std::string &workid) : audio_flage_(false)
    {
        pcmdata = buffer_create();
        _ax_init();
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
        if (axera_session_) axera_session_->Release();
        _ax_deinit();
    }
};

int llm_task::ax_init_flage_ = 0;

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
        for (int i = 0; i < (int)wav_data.size() - 4; i++) {
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
                auto p = _llm_task_obj.lock();
                if (p) p->sys_pcm_on_data(raw->string());
            });
            llm_task_obj->audio_flage_ = true;
        }
    }

    void work(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_kws::work:%s", data.c_str());
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
        SLOGI("llm_kws::pause:%s", data.c_str());
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

    void task_buttons_data(const std::weak_ptr<llm_task> llm_task_obj_weak,
                           const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                           const std::string &data)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        if (data.empty() || (data == "None")) return;

        try {
            std::string user_msg    = sample_unescapeString(data);
            nlohmann::json btn_json = nlohmann::json::parse(user_msg);

            if (btn_json.contains("code") && btn_json.contains("vale")) {
                int current_code = btn_json["code"];
                int current_vale = btn_json["vale"];

                if (current_vale == 204) {
                    int last_code = llm_task_obj->get_btn_204_state();

                    if (last_code == 0 && current_code == 1) {
                        llm_task_obj->trigger_wakeup();
                    }

                    llm_task_obj->set_btn_204_state(current_code);
                }
            }
        } catch (const std::exception &e) {
            SLOGE("Button data JSON parse error: %s", e.what());
        }
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
            llm_channel->set_output(llm_task_obj->enoutput());
            llm_channel->set_stream(llm_task_obj->enstream_flag());
            llm_task_obj->play_awake_wav = std::bind(&llm_kws::play_awake_wav, this, std::placeholders::_1);
            llm_task_obj->set_output(std::bind(&llm_kws::task_output, this, _llm_task_obj,
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));
            for (const auto &input : llm_task_obj->inputs()) {
                if (input.find("sys") != std::string::npos) {
                    audio_url_ = unit_call("audio", "cap", "None");
                    llm_channel->subscriber(audio_url_,
                                            [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                                                auto p = _llm_task_obj.lock();
                                                if (p) p->sys_pcm_on_data(raw->string());
                                            });
                    llm_task_obj->audio_flage_ = true;
                } else if (input.find("kws") != std::string::npos) {
                    llm_task_obj->delay_audio_frame_ = 0;
                    llm_channel->subscriber_work_id("", std::bind(&llm_kws::task_user_data, this, _llm_task_obj,
                                                                  std::weak_ptr<llm_channel_obj>(llm_channel),
                                                                  std::placeholders::_1, std::placeholders::_2));
                } else if (input.find("buttons_thread") != std::string::npos) {
                    std::string socket_url = "ipc:///tmp/llm/ec_prox.event.socket";
                    auto business_logic    = std::bind(
                        &llm_kws::task_buttons_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                        std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2);

                    llm_channel->subscriber(
                        socket_url, [llm_channel, business_logic](StackFlows::pzmq *p,
                                                                  const std::shared_ptr<StackFlows::pzmq_data> &d) {
                            llm_channel->subscriber_event_call(business_logic, p, d);
                        });
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

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_kws::link:%s", data.c_str());
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
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                if (auto p = _llm_task_obj.lock()) p->sys_pcm_on_data(raw->string());
            });

            llm_task_obj->audio_flage_ = true;
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("buttons_thread") != std::string::npos) {
            std::string socket_url = "ipc:///tmp/llm/ec_prox.event.socket";
            auto business_logic =
                std::bind(&llm_kws::task_buttons_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2);

            llm_channel->subscriber(
                socket_url,
                [llm_channel, business_logic](StackFlows::pzmq *p, const std::shared_ptr<StackFlows::pzmq_data> &d) {
                    llm_channel->subscriber_event_call(business_logic, p, d);
                });

            llm_task_obj->inputs_.push_back(data);
        } else {
            error_body["code"]    = -22;
            error_body["message"] = "unsupported link target";
            send("None", "None", error_body, work_id);
            return;
        }

        if (ret) {
            error_body["code"]    = -20;
            error_body["message"] = "link false";
            send("None", "None", error_body, work_id);
            return;
        }
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    void unlink(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_kws::unlink:%s", data.c_str());
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

        llm_channel->stop_subscriber_work_id(data);

        for (auto it = llm_task_obj->inputs_.begin(); it != llm_task_obj->inputs_.end();) {
            if (*it == data)
                it = llm_task_obj->inputs_.erase(it);
            else
                ++it;
        }

        if (data.find("sys") != std::string::npos) {
            llm_task_obj->audio_flage_ = false;
        }

        send("None", "None", LLM_NO_ERROR, work_id);
    }

    void taskinfo(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_kws::taskinfo:%s", data.c_str());
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto &task_channel) { return task_channel.second->work_id_; });
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
            req_body["model"]           = llm_task_obj->model();
            req_body["response_format"] = llm_task_obj->response_format();
            req_body["enoutput"]        = llm_task_obj->enoutput();
            req_body["inputs"]          = llm_task_obj->inputs();
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
        llm_task_[work_id_num]->trigger_wakeup();
        return LLM_NONE;
    }

    ~llm_kws()
    {
        while (1) {
            auto it = llm_task_.begin();
            if (it == llm_task_.end()) {
                break;
            }
            it->second->stop();
            if (it->second->audio_flage_) {
                unit_call("audio", "cap_stop", "None");
            }
            get_channel(it->first)->stop_subscriber("");
            it->second.reset();
            llm_task_.erase(it->first);
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