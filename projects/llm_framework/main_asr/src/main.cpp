/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "sherpa-ncnn/csrc/recognizer.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

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

#define NCNN_ASR_CONFIG_AUTO_SET(obj, key)    \
    if (config_body.contains(#key))           \
        ncnn_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        ncnn_config_.key = obj[#key];

#define ONNX_ONLINE_CONFIG_AUTO_SET(obj, key)                \
    if (config_body.contains(#key))                          \
        config_body.at(#key).get_to(onnx_online_config.key); \
    else if ((obj).contains(#key))                           \
        (obj).at(#key).get_to(onnx_online_config.key);

#define ONNX_ASR_CONFIG_AUTO_SET(obj, key)        \
    if (config_body.contains(#key))               \
        onnx_asr_config_.key = config_body[#key]; \
    else if (obj.contains(#key))                  \
        onnx_asr_config_.key = obj[#key];

#define ONNX_VAD_CONFIG_AUTO_SET(obj, key)   \
    if (config_body.contains(#key))          \
        vad_config_.key = config_body[#key]; \
    else if (obj.contains(#key))             \
        vad_config_.key = obj[#key];

class llm_task {
private:
    sherpa_ncnn::RecognizerConfig ncnn_config_;
    std::unique_ptr<sherpa_ncnn::Recognizer> ncnn_recognizer_;
    std::unique_ptr<sherpa_ncnn::Stream> ncnn_stream_;

    sherpa_onnx::OfflineRecognizerConfig onnx_asr_config_;
    sherpa_onnx::OnlineRecognizerConfig onnx_online_config;

    sherpa_onnx::VadModelConfig vad_config_;
    std::unique_ptr<sherpa_onnx::OfflineStream> offline_stream_;
    std::unique_ptr<sherpa_onnx::OfflineRecognizer> onnx_recognizer_;
    std::unique_ptr<sherpa_onnx::OnlineRecognizer> onnx_online_recognizer_;
    std::unique_ptr<sherpa_onnx::OnlineStream> online_stream;
    std::unique_ptr<sherpa_onnx::VoiceActivityDetector> vad_;

    enum EngineType {
        ENGINE_NCNN   = 0,
        ENGINE_ONNX   = 1,
        ENGINE_ONLINE = 3,
    } engine_type_ = ENGINE_NCNN;

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
    int awake_delay_        = 50;
    int delay_audio_frame_  = 10;
    float silence_ms_accum_ = 0.0f;
    float silence_timeout   = 1000.0f;

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

        if (model_.rfind("sherpa-ncnn", 0) == 0) {
            engine_type_ = ENGINE_NCNN;
        } else if (model_.rfind("sherpa-onnx", 0) == 0) {
            engine_type_ = ENGINE_ONLINE;
        } else {
            engine_type_ = ENGINE_ONNX;
        }

        enstream_ = response_format_.find("stream") == std::string::npos ? false : true;
        return false;
    }

    int load_ncnn_model(const nlohmann::json &config_body, const nlohmann::json &file_body)
    {
        std::string base_model = base_model_path_ + model_ + "/";
        SLOGI("base_model (ncnn) %s", base_model.c_str());

        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.sampling_rate);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.feature_dim);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.encoder_param);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.encoder_bin);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.decoder_param);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.decoder_bin);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.joiner_param);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.joiner_bin);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.tokens);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.encoder_opt.num_threads);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.decoder_opt.num_threads);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.joiner_opt.num_threads);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], decoder_config.method);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], decoder_config.num_active_paths);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.must_contain_nonsilence);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.min_trailing_silence);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.min_utterance_length);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.must_contain_nonsilence);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.min_trailing_silence);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.min_utterance_length);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.must_contain_nonsilence);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.min_trailing_silence);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.min_utterance_length);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], enable_endpoint);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], hotwords_file);
        NCNN_ASR_CONFIG_AUTO_SET(file_body["mode_param"], hotwords_score);

        if (config_body.contains("awake_delay"))
            awake_delay_ = config_body["awake_delay"].get<int>();
        else if (file_body["mode_param"].contains("awake_delay"))
            awake_delay_ = file_body["mode_param"]["awake_delay"];

        if (config_body.contains("rule1")) {
            ncnn_config_.endpoint_config.rule1.min_trailing_silence = config_body["rule1"].get<float>();
            ncnn_config_.endpoint_config.rule1.must_contain_nonsilence =
                (ncnn_config_.endpoint_config.rule1.min_trailing_silence == 0.0f) ? false : true;
        }
        if (config_body.contains("rule2")) {
            ncnn_config_.endpoint_config.rule2.min_trailing_silence = config_body["rule2"].get<float>();
            ncnn_config_.endpoint_config.rule2.must_contain_nonsilence =
                (ncnn_config_.endpoint_config.rule2.min_trailing_silence == 0.0f) ? false : true;
        }
        if (config_body.contains("rule3")) {
            ncnn_config_.endpoint_config.rule3.min_utterance_length = config_body["rule3"].get<float>();
            ncnn_config_.endpoint_config.rule3.must_contain_nonsilence =
                (ncnn_config_.endpoint_config.rule3.min_utterance_length == 0.0f) ? false : true;
        }

        ncnn_config_.model_config.tokens        = base_model + ncnn_config_.model_config.tokens;
        ncnn_config_.model_config.encoder_param = base_model + ncnn_config_.model_config.encoder_param;
        ncnn_config_.model_config.encoder_bin   = base_model + ncnn_config_.model_config.encoder_bin;
        ncnn_config_.model_config.decoder_param = base_model + ncnn_config_.model_config.decoder_param;
        ncnn_config_.model_config.decoder_bin   = base_model + ncnn_config_.model_config.decoder_bin;
        ncnn_config_.model_config.joiner_param  = base_model + ncnn_config_.model_config.joiner_param;
        ncnn_config_.model_config.joiner_bin    = base_model + ncnn_config_.model_config.joiner_bin;

        ncnn_recognizer_ = std::make_unique<sherpa_ncnn::Recognizer>(ncnn_config_);
        return 0;
    }

    int load_onnx_model(const nlohmann::json &config_body, const nlohmann::json &file_body)
    {
        std::string base_model = base_model_path_ + model_ + "/";
        SLOGI("base_model (onnx) %s", base_model.c_str());

        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.sampling_rate);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.feature_dim);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.low_freq);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.dither);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.normalize_samples);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.snip_edges);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.frame_shift_ms);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.frame_length_ms);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_librosa);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.remove_dc_offset);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.preemph_coeff);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.window_type);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.nemo_normalize_type);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.num_ceps);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.use_energy);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_mfcc);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_whisper);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_t_one);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.round_to_power_of_two);

        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.encoder_filename);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.decoder_filename);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.joiner_filename);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.paraformer.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.nemo_ctc.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.whisper.encoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.whisper.decoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.whisper.language);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.whisper.task);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.whisper.tail_paddings);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.fire_red_asr.encoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.fire_red_asr.decoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.tdnn.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_ctc.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.wenet_ctc.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.sense_voice.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.sense_voice.language);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.sense_voice.use_itn);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.moonshine.preprocessor);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.moonshine.encoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.moonshine.uncached_decoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.moonshine.cached_decoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.dolphin.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.canary.encoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.canary.decoder);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.canary.src_lang);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.canary.tgt_lang);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.canary.use_pnc);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.omnilingual.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.telespeech_ctc);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.tokens);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.num_threads);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.debug);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.model_type);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.modeling_unit);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], model_config.bpe_vocab);

        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.model);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.scale);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lm_num_threads);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lm_provider);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lodr_fst);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lodr_scale);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lodr_backoff_id);

        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], ctc_fst_decoder_config.graph);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], ctc_fst_decoder_config.max_active);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], decoding_method);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], max_active_paths);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], hotwords_file);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], hotwords_score);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], blank_penalty);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], rule_fsts);
        ONNX_ASR_CONFIG_AUTO_SET(file_body["mode_param"], rule_fars);

        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.model);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.threshold);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.min_silence_duration);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.min_speech_duration);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.window_size);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], silero_vad.max_speech_duration);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], ten_vad.model);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], ten_vad.threshold);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], ten_vad.min_silence_duration);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], ten_vad.min_speech_duration);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], ten_vad.window_size);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], ten_vad.max_speech_duration);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], sample_rate);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], num_threads);
        ONNX_VAD_CONFIG_AUTO_SET(file_body["mode_param"], debug);

        if (config_body.contains("awake_delay"))
            awake_delay_ = config_body["awake_delay"].get<int>();
        else if (file_body["mode_param"].contains("awake_delay"))
            awake_delay_ = file_body["mode_param"]["awake_delay"];

        if (config_body.contains("silence_timeout"))
            silence_timeout = config_body["silence_timeout"].get<int>();
        else if (file_body["mode_param"].contains("silence_timeout"))
            silence_timeout = file_body["mode_param"]["silence_timeout"];

        onnx_asr_config_.model_config.sense_voice.model = base_model + onnx_asr_config_.model_config.sense_voice.model;
        onnx_asr_config_.model_config.tokens            = base_model + onnx_asr_config_.model_config.tokens;
        vad_config_.silero_vad.model                    = base_model + vad_config_.silero_vad.model;

        onnx_recognizer_ = std::make_unique<sherpa_onnx::OfflineRecognizer>(onnx_asr_config_);
        vad_             = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config_);
        return 0;
    }

    int load_online_model(const nlohmann::json &config_body, const nlohmann::json &file_body)
    {
        std::string base_model = base_model_path_ + model_ + "/";
        SLOGI("base_model (onnx) %s", base_model.c_str());

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.sampling_rate);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.feature_dim);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.low_freq);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.high_freq);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.dither);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.normalize_samples);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.snip_edges);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.frame_shift_ms);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.frame_length_ms);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_librosa);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.remove_dc_offset);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.preemph_coeff);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.window_type);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.nemo_normalize_type);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.num_ceps);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.use_energy);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_mfcc);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_whisper);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.is_t_one);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], feat_config.round_to_power_of_two);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.encoder);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.decoder);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.transducer.joiner);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.provider_config.provider);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.paraformer.encoder);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.paraformer.decoder);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.wenet_ctc.model);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.wenet_ctc.chunk_size);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.wenet_ctc.num_left_chunks);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer2_ctc.model);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.nemo_ctc.model);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.t_one_ctc.model);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.encoder_dims);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.attention_dims);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.num_encoder_layers);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.cnn_module_kernels);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.left_context_len);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.T);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.decode_chunk_len);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.zipformer_meta.context_size);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.tokens);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.num_threads);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.warm_up);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.model_type);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.modeling_unit);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.bpe_vocab);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], model_config.tokens_buf);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.model);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.scale);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lm_num_threads);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lm_provider);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lodr_fst);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lodr_scale);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.lodr_backoff_id);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], lm_config.shallow_fusion);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.must_contain_nonsilence);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.min_trailing_silence);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule1.min_utterance_length);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.must_contain_nonsilence);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.min_trailing_silence);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule2.min_utterance_length);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.must_contain_nonsilence);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.min_trailing_silence);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], endpoint_config.rule3.min_utterance_length);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], ctc_fst_decoder_config.graph);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], ctc_fst_decoder_config.max_active);

        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], enable_endpoint);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], decoding_method);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], max_active_paths);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], hotwords_file);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], hotwords_score);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], blank_penalty);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], temperature_scale);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], rule_fsts);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], rule_fars);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], reset_encoder);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], hr.dict_dir);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], hr.lexicon);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], hr.rule_fsts);
        ONNX_ONLINE_CONFIG_AUTO_SET(file_body["mode_param"], hotwords_buf);

        if (config_body.contains("awake_delay"))
            awake_delay_ = config_body["awake_delay"].get<int>();
        else if (file_body["mode_param"].contains("awake_delay"))
            awake_delay_ = file_body["mode_param"]["awake_delay"];

        onnx_online_config.model_config.transducer.encoder =
            base_model + onnx_online_config.model_config.transducer.encoder;
        onnx_online_config.model_config.transducer.decoder =
            base_model + onnx_online_config.model_config.transducer.decoder;
        onnx_online_config.model_config.transducer.joiner =
            base_model + onnx_online_config.model_config.transducer.joiner;
        onnx_online_config.model_config.tokens = base_model + onnx_online_config.model_config.tokens;

        onnx_online_recognizer_ = std::make_unique<sherpa_onnx::OnlineRecognizer>(onnx_online_config);

        return 0;
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

            if (engine_type_ == ENGINE_NCNN) {
                return load_ncnn_model(config_body, file_body);
            } else if (engine_type_ == ENGINE_ONLINE) {
                return load_online_model(config_body, file_body);
            } else {
                return load_onnx_model(config_body, file_body);
            }
        } catch (...) {
            SLOGE("config false");
            return -3;
        }
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

    void sys_pcm_on_data_ncnn(const std::string &raw)
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
        int16_t audio_val;
        while (buffer_read_i16(pcmdata, &audio_val, 1)) {
            float normalizedSample = static_cast<float>(audio_val) / INT16_MAX;
            floatSamples.push_back(normalizedSample);
        }
        buffer_resize(pcmdata, 0);
        count = 0;

        if (awake_flage_ && ncnn_stream_) {
            ncnn_stream_.reset();
            awake_flage_ = false;
        }
        if (!ncnn_stream_) {
            ncnn_stream_ = ncnn_recognizer_->CreateStream();
        }

        ncnn_stream_->AcceptWaveform(ncnn_config_.feat_config.sampling_rate, floatSamples.data(), floatSamples.size());

        while (ncnn_recognizer_->IsReady(ncnn_stream_.get())) {
            ncnn_recognizer_->DecodeStream(ncnn_stream_.get());
        }

        std::string text = ncnn_recognizer_->GetResult(ncnn_stream_.get()).text;
        std::string lower_text;
        lower_text.resize(text.size());
        std::transform(text.begin(), text.end(), lower_text.begin(), [](const char c) { return std::tolower(c); });

        if ((!lower_text.empty()) && out_callback_) out_callback_(lower_text, false);

        bool is_endpoint = ncnn_recognizer_->IsEndpoint(ncnn_stream_.get());
        if (is_endpoint) {
            ncnn_stream_->Finalize();
            if ((!lower_text.empty()) && out_callback_) {
                out_callback_(lower_text, true);
            }
            ncnn_stream_.reset();
            if (ensleep_) {
                if (pause) pause();
            }
        }
    }

    void sys_pcm_on_data_onnx(const std::string &raw)
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
        int16_t audio_val;
        while (buffer_read_i16(pcmdata, &audio_val, 1)) {
            float normalizedSample = static_cast<float>(audio_val) / INT16_MAX;
            floatSamples.push_back(normalizedSample);
        }
        buffer_resize(pcmdata, 0);
        count = 0;

        vad_->AcceptWaveform(floatSamples.data(), floatSamples.size());
        while (!vad_->Empty()) {
            const auto &segment = vad_->Front();
            if (!offline_stream_) offline_stream_ = onnx_recognizer_->CreateStream();
            offline_stream_->AcceptWaveform(onnx_asr_config_.feat_config.sampling_rate, segment.samples.data(),
                                            segment.samples.size());
            onnx_recognizer_->DecodeStream(offline_stream_.get());
            const auto &result = offline_stream_->GetResult();
            if (!result.text.empty() && out_callback_) {
                out_callback_(result.text, true);
            }
            vad_->Pop();
            offline_stream_.reset();
        }

        {
            bool detected  = vad_->IsSpeechDetected();
            float chunk_ms = (delay_audio_frame_ + 1) * 10.0f;

            if (detected) {
                silence_ms_accum_ = 0.0f;
            } else {
                silence_ms_accum_ += chunk_ms;
            }
            if (silence_ms_accum_ >= silence_timeout) {
                if (ensleep_) {
                    if (pause) pause();
                }
                silence_ms_accum_ = 0.0f;
            }
        }
    }

    void sys_pcm_on_data_online(const std::string &raw)
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
        int16_t audio_val;
        while (buffer_read_i16(pcmdata, &audio_val, 1)) {
            float normalizedSample = static_cast<float>(audio_val) / INT16_MAX;
            floatSamples.push_back(normalizedSample);
        }
        buffer_resize(pcmdata, 0);
        count = 0;

        if (!online_stream) online_stream = onnx_online_recognizer_->CreateStream();
        online_stream->AcceptWaveform(onnx_online_config.feat_config.sampling_rate, floatSamples.data(),
                                      floatSamples.size());

        while (onnx_online_recognizer_->IsReady(online_stream.get())) {
            onnx_online_recognizer_->DecodeStream(online_stream.get());
        }

        auto text = onnx_online_recognizer_->GetResult(online_stream.get()).text;
        std::string lower_text;
        lower_text.resize(text.size());
        std::transform(text.begin(), text.end(), lower_text.begin(), [](auto c) { return std::tolower(c); });

        if ((!lower_text.empty()) && out_callback_) out_callback_(lower_text, false);

        bool is_endpoint = onnx_online_recognizer_->IsEndpoint(online_stream.get());

        if (is_endpoint) {
            if ((!lower_text.empty()) && out_callback_) {
                out_callback_(lower_text, true);
            }
            online_stream.reset();
            if (ensleep_) {
                if (pause) pause();
            }
        }
    }

    void sys_pcm_on_data(const std::string &raw)
    {
        if (engine_type_ == ENGINE_NCNN) {
            sys_pcm_on_data_ncnn(raw);
        } else if (engine_type_ == ENGINE_ONLINE) {
            sys_pcm_on_data_online(raw);
        } else {
            sys_pcm_on_data_onnx(raw);
        }
    }

    void kws_awake()
    {
        awake_flage_ = true;
    }

    bool delete_model()
    {
        ncnn_recognizer_.reset();
        onnx_recognizer_.reset();
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
        event_queue_.appendListener(EVENT_TASK_PAUSE, std::bind(&llm_asr::_task_pause, this, std::placeholders::_1));
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
            for (int i = 0; i < (int)in.length() - 4; i++) {
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
                }
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
        llm_task_obj->kws_awake();
        if ((!audio_url_.empty()) && (llm_task_obj->audio_flage_ == false)) {
            std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                if (auto p = _llm_task_obj.lock()) {
                    p->sys_pcm_on_data(raw->string());
                }
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
        SLOGI("llm_asr::pause:%s", data.c_str());
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
            send("None", "None", error_body, "asr");
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

            for (const auto &input : llm_task_obj->inputs_) {
                if (input.find("sys") != std::string::npos) {
                    audio_url_                            = unit_call("audio", "cap", input);
                    std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
                    llm_channel->subscriber(audio_url_,
                                            [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                                                if (auto p = _llm_task_obj.lock()) {
                                                    p->sys_pcm_on_data(raw->string());
                                                }
                                            });
                    llm_task_obj->audio_flage_ = true;
                } else if (input.find("asr") != std::string::npos) {
                    llm_task_obj->delay_audio_frame_ = 0;
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
            SLOGI("load_model success");
            send("None", "None", LLM_NO_ERROR, work_id);
            return 0;
        } else {
            SLOGE("load_model Failed");
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
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                if (auto p = _llm_task_obj.lock()) {
                    p->sys_pcm_on_data(raw->string());
                }
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
                           [](const auto &task_channel) { return task_channel.second->work_id_; });
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
    llm_asr llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}