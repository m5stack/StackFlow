/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "OnnxWrapper.hpp"
#include "EngineWrapper.hpp"
#include "Lexicon.hpp"
#include <ax_sys_api.h>
#include "AudioFile.h"

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string.h>
#include <samplerate.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "subprocess.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_melotts will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;

typedef struct {
    std::string mode;
    std::string encoder;
    std::string decoder;
    std::string lexicon;
    std::string tokens;
    std::string gbin;
    std::string sentence;
    std::string tagger;
    std::string verbalizer;
    float spacker_speed = 1.0;
    int mode_rate       = 44100;
    int audio_rate      = 16000;
    int spacker_role    = 0;
    float noise_scale   = 0.3f;
    float length_scale  = 1.0;
    float noise_scale_w = 0.6f;
    float sdp_ratio     = 0.2f;

    float get_length_scale()
    {
        return (float)(length_scale / spacker_speed);
    }
} melotts_config;

typedef std::function<void(const std::string &data, bool finish)> task_callback_t;

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

class llm_task {
private:
public:
    melotts_config mode_config_;
    std::unique_ptr<OnnxWrapper> encoder_;
    std::unique_ptr<EngineWrapper> decoder_;
    std::unique_ptr<Lexicon> lexicon_;
    std::vector<float> g_matrix;
    std::string model_;
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
    std::string tts_string_stream_buff;

    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            model_           = config_body.at("model");
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");
            if (config_body.contains("enaudio")) enaudio_ = config_body.at("enaudio");
            if (config_body.contains("input")) {
                if (config_body["input"].is_string()) {
                    inputs_.push_back(config_body["input"].get<std::string>());
                } else if (config_body["input"].is_array()) {
                    for (auto _in : config_body["input"]) {
                        inputs_.push_back(_in.get<std::string>());
                    }
                }
            } else
                throw std::string("error");
        } catch (...) {
            SLOGE("setup config_body error");
            return true;
        }
        enstream_ = response_format_.find("stream") == std::string::npos ? false : true;
        return false;
    }

    std::unordered_map<std::string, int> MELOTTS_LANG_IDS_MAP{
        {"melotts-ja-jp", 1}, {"melotts-en-us", 2}, {"melotts_zh-cn", 3}, {"melotts-zh-cn", 3}};

    std::vector<int> intersperse(const std::vector<int> &lst, int item)
    {
        std::vector<int> result(lst.size() * 2 + 1, item);
        for (size_t i = 1; i < result.size(); i += 2) {
            result[i] = lst[i / 2];
        }
        return result;
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
        if (model_ == "melotts_zh-cn")
            config_file_paths = get_config_file_paths(base_model_path_, base_model_config_path_, "melotts-zh-cn");

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
            CONFIG_AUTO_SET(file_body["mode_param"], tokens);
            CONFIG_AUTO_SET(file_body["mode_param"], lexicon);
            CONFIG_AUTO_SET(file_body["mode_param"], sentence);
            CONFIG_AUTO_SET(file_body["mode_param"], spacker_role);
            CONFIG_AUTO_SET(file_body["mode_param"], mode_rate);
            CONFIG_AUTO_SET(file_body["mode_param"], audio_rate);
            CONFIG_AUTO_SET(file_body["mode_param"], spacker_speed);
            CONFIG_AUTO_SET(file_body["mode_param"], gbin);
            CONFIG_AUTO_SET(file_body["mode_param"], encoder);
            CONFIG_AUTO_SET(file_body["mode_param"], decoder);
            CONFIG_AUTO_SET(file_body["mode_param"], noise_scale);
            CONFIG_AUTO_SET(file_body["mode_param"], length_scale);
            CONFIG_AUTO_SET(file_body["mode_param"], noise_scale_w);
            CONFIG_AUTO_SET(file_body["mode_param"], sdp_ratio);
            CONFIG_AUTO_SET(file_body["mode_param"], tagger);
            CONFIG_AUTO_SET(file_body["mode_param"], verbalizer);
            mode_config_.tokens     = base_model + mode_config_.tokens;
            mode_config_.gbin       = base_model + mode_config_.gbin;
            mode_config_.encoder    = base_model + mode_config_.encoder;
            mode_config_.decoder    = base_model + mode_config_.decoder;
            mode_config_.lexicon    = base_model + mode_config_.lexicon;
            mode_config_.tagger     = base_model + mode_config_.tagger;
            mode_config_.verbalizer = base_model + mode_config_.verbalizer;
            if (config_body.contains("awake_delay"))
                awake_delay_ = config_body["awake_delay"].get<int>();
            else if (file_body["mode_param"].contains("awake_delay"))
                awake_delay_ = file_body["mode_param"]["awake_delay"];
            // Load lexicon
            lexicon_ = std::make_unique<Lexicon>(mode_config_.lexicon, mode_config_.tokens, mode_config_.tagger,
                                                 mode_config_.verbalizer);
            // Read g.bin
            g_matrix.resize(256, 0);
            FILE *fp = fopen(mode_config_.gbin.c_str(), "rb");
            if (!fp) {
                printf("Open %s failed!\n", mode_config_.gbin.c_str());
                return -3;
            }
            fread(g_matrix.data(), sizeof(float), g_matrix.size(), fp);
            fclose(fp);
            encoder_ = std::make_unique<OnnxWrapper>();
            decoder_ = std::make_unique<EngineWrapper>();
            if (0 != encoder_->Init(mode_config_.encoder)) {
                printf("encoder init failed!\n");
                return -4;
            }
            if (0 != decoder_->Init(mode_config_.decoder.c_str())) {
                printf("Init decoder model failed!\n");
                return -5;
            }
        } catch (...) {
            SLOGE("config false");
            return -6;
        }
        return 0;
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

    void resample_audio(float *input_buffer, int input_length, float *output_buffer, int *output_length,
                        double src_ratio)
    {
        SRC_STATE *src_state;
        int error;
        src_state = src_new(SRC_SINC_FASTEST, 1, &error);
        if (!src_state) {
            fprintf(stderr, "Error : src_new() failed: %s\n", src_strerror(error));
            throw std::string("src_new() failed");
        }
        SRC_DATA src_data;
        src_data.data_in       = input_buffer;
        src_data.input_frames  = input_length;
        src_data.src_ratio     = src_ratio;
        int max_output_length  = (int)(input_length * src_ratio + 1);
        src_data.data_out      = output_buffer;
        src_data.output_frames = max_output_length;
        error                  = src_process(src_state, &src_data);
        if (error) {
            fprintf(stderr, "Error : src_process() failed: %s\n", src_strerror(error));
            src_delete(src_state);
            throw std::string("src_process() failed");
        }
        *output_length = src_data.output_frames_gen;
        src_delete(src_state);
    }

    bool TTS(const std::string &msg_str, bool finish)
    {
        try {
            std::vector<int16_t> wav_pcm_data;
            if (msg_str.empty()) {
                if (out_callback_) {
                    std::string output = wav_pcm_data.empty() ? std::string()
                                                              : std::string((char *)wav_pcm_data.data(),
                                                                            wav_pcm_data.size() * sizeof(int16_t));
                    out_callback_(output, finish);
                }
                return false;
            }

            std::vector<int> phones_bef, tones_bef;
            lexicon_->convert(msg_str, phones_bef, tones_bef);
            auto phones   = intersperse(phones_bef, 0);
            auto tones    = intersperse(tones_bef, 0);
            int phone_len = phones.size();
            std::vector<int> langids(phone_len, 3);

            auto encoder_output =
                encoder_->Run(phones, tones, langids, g_matrix, mode_config_.noise_scale, mode_config_.noise_scale_w,
                              mode_config_.get_length_scale(), mode_config_.sdp_ratio);
            float *zp_data = encoder_output.at(0).GetTensorMutableData<float>();
            int audio_len  = encoder_output.at(2).GetTensorMutableData<int>()[0];
            auto zp_info   = encoder_output.at(0).GetTensorTypeAndShapeInfo();
            auto zp_shape  = zp_info.GetShape();

            int zp_size         = decoder_->GetInputSize(0) / sizeof(float);
            int dec_len         = zp_size / zp_shape[1];
            int audio_slice_len = decoder_->GetOutputSize(0) / sizeof(float);

            const int pad_frames        = 24;
            const int samples_per_frame = 512;

            const int effective_frames = dec_len - 2 * pad_frames;

            int dec_slice_num =
                static_cast<int>(std::ceil(static_cast<double>(zp_shape[2]) / static_cast<double>(effective_frames)));

            const int sola_buffer_frame = pad_frames * samples_per_frame;
            const int sola_search_frame = pad_frames * samples_per_frame;
            const int block_frame       = (dec_len - 2 * pad_frames) * samples_per_frame;

            std::vector<float> fade_in_window(sola_buffer_frame);
            std::vector<float> fade_out_window(sola_buffer_frame);

            for (int i = 0; i < sola_buffer_frame; i++) {
                fade_in_window[i]  = static_cast<float>(i) / sola_buffer_frame;
                fade_out_window[i] = 1.0f - fade_in_window[i];
            }

            std::vector<float> sola_buffer(sola_buffer_frame, 0.0f);
            bool first_frame = true;

            std::vector<float> pcmlist;

            for (int i = 0; i < dec_slice_num; i++) {
                int input_start = i * effective_frames;
                if (i > 0) {
                    input_start -= pad_frames;
                }
                input_start = std::max(0, input_start);

                int actual_len = std::min(dec_len, static_cast<int>(zp_shape[2] - input_start));

                int output_start_frame, output_end_frame;

                if (i == 0) {
                    output_start_frame = 0;
                    output_end_frame   = effective_frames - 1;
                } else if (i == dec_slice_num - 1) {
                    output_start_frame = i * effective_frames;
                    output_end_frame   = static_cast<int>(zp_shape[2]) - 1;
                } else {
                    output_start_frame = i * effective_frames;
                    output_end_frame   = (i + 1) * effective_frames - 1;
                }

                std::vector<float> zp(zp_size, 0);

                for (int n = 0; n < zp_shape[1]; n++) {
                    int copy_size = std::min(actual_len, static_cast<int>(zp_shape[2] - input_start));
                    if (copy_size > 0) {
                        memcpy(zp.data() + n * dec_len, zp_data + n * zp_shape[2] + input_start,
                               sizeof(float) * copy_size);
                    }
                }

                std::vector<float> decoder_output(audio_slice_len);
                decoder_->SetInput(zp.data(), 0);
                decoder_->SetInput(g_matrix.data(), 1);

                if (0 != decoder_->Run()) {
                    throw std::string("decoder_ RunSync error");
                }

                decoder_->GetOutput(decoder_output.data(), 0);

                if (first_frame) {
                    int audio_start = 0;
                    int audio_len   = decoder_output.size() - sola_buffer_frame;
                    audio_len       = std::max(0, audio_len);

                    if (audio_len > 0) {
                        pcmlist.insert(pcmlist.end(), decoder_output.begin() + audio_start,
                                       decoder_output.begin() + audio_start + audio_len);
                    }

                    int buffer_start = audio_len;

                    if (buffer_start + sola_buffer_frame <= decoder_output.size()) {
                        std::copy(decoder_output.begin() + buffer_start,
                                  decoder_output.begin() + buffer_start + sola_buffer_frame, sola_buffer.begin());
                    } else {
                        int available = static_cast<int>(decoder_output.size() - buffer_start);
                        if (available > 0) {
                            std::copy(decoder_output.begin() + buffer_start, decoder_output.end(), sola_buffer.begin());
                            std::fill(sola_buffer.begin() + available, sola_buffer.end(), 0.0f);
                        } else {
                            std::fill(sola_buffer.begin(), sola_buffer.end(), 0.0f);
                        }
                    }

                    first_frame = false;

                } else {
                    int audio_start = pad_frames * samples_per_frame;

                    std::vector<float> search_window(sola_buffer_frame + sola_search_frame);
                    std::copy(decoder_output.begin() + audio_start,
                              decoder_output.begin() + audio_start + search_window.size(), search_window.begin());

                    int best_offset        = 0;
                    float best_correlation = -1.0;

                    for (int offset = 0; offset <= sola_search_frame; offset++) {
                        float correlation = 0.0;
                        float energy      = 0.0;

                        for (int j = 0; j < sola_buffer_frame; j++) {
                            correlation += sola_buffer[j] * search_window[j + offset];
                            energy += search_window[j + offset] * search_window[j + offset];
                        }

                        float normalized_correlation = (energy > 1e-8) ? correlation / std::sqrt(energy) : 0.0f;

                        if (normalized_correlation > best_correlation) {
                            best_correlation = normalized_correlation;
                            best_offset      = offset;
                        }
                    }


                    int aligned_start = audio_start + best_offset;

                    std::vector<float> crossfade_region(sola_buffer_frame);

                    for (int j = 0; j < sola_buffer_frame; j++) {
                        crossfade_region[j] =
                            decoder_output[aligned_start + j] * fade_in_window[j] + sola_buffer[j] * fade_out_window[j];
                    }

                    pcmlist.insert(pcmlist.end(), crossfade_region.begin(), crossfade_region.end());

                    int remaining_start = aligned_start + sola_buffer_frame;

                    if (i == dec_slice_num - 1) {
                        int total_expected_samples = audio_len * samples_per_frame / 512;
                        int processed_samples      = static_cast<int>(pcmlist.size());
                        int remaining_needed       = total_expected_samples - processed_samples;
                        remaining_needed           = std::max(0, remaining_needed);

                        int remaining_len =
                            std::min(remaining_needed, static_cast<int>(decoder_output.size() - remaining_start));

                        if (remaining_len > 0) {
                            pcmlist.insert(pcmlist.end(), decoder_output.begin() + remaining_start,
                                           decoder_output.begin() + remaining_start + remaining_len);
                        }

                    } else {
                        int remaining_len = (dec_len - 2 * pad_frames) * samples_per_frame - sola_buffer_frame;
                        remaining_len =
                            std::min(remaining_len, static_cast<int>(decoder_output.size() - remaining_start));

                        if (remaining_len > 0) {
                            pcmlist.insert(pcmlist.end(), decoder_output.begin() + remaining_start,
                                           decoder_output.begin() + remaining_start + remaining_len);
                        }

                        int buffer_start = remaining_start + remaining_len;

                        if (buffer_start + sola_buffer_frame <= decoder_output.size()) {
                            std::copy(decoder_output.begin() + buffer_start,
                                      decoder_output.begin() + buffer_start + sola_buffer_frame, sola_buffer.begin());
                        } else {
                            int avail = static_cast<int>(decoder_output.size() - buffer_start);
                            if (avail > 0) {
                                std::copy(decoder_output.begin() + buffer_start, decoder_output.end(),
                                          sola_buffer.begin());
                            }
                            std::fill(sola_buffer.begin() + avail, sola_buffer.end(), 0.0f);
                        }
                    }
                }
            }

            if (pcmlist.size() > audio_len) {
                pcmlist.resize(audio_len);
            }


            double src_ratio =
                static_cast<double>(mode_config_.audio_rate) / static_cast<double>(mode_config_.mode_rate);
            std::vector<float> tmp_pcm((pcmlist.size() * src_ratio + 1));
            int len;

            resample_audio(pcmlist.data(), pcmlist.size(), tmp_pcm.data(), &len, src_ratio);


            wav_pcm_data.reserve(len);
            std::transform(tmp_pcm.begin(), tmp_pcm.begin() + len, std::back_inserter(wav_pcm_data),
                           [](const auto val) { return static_cast<int16_t>(val * INT16_MAX); });

            if (out_callback_) {
                out_callback_(
                    std::string(reinterpret_cast<char *>(wav_pcm_data.data()), wav_pcm_data.size() * sizeof(int16_t)),
                    finish);
            }

        } catch (const std::exception &e) {
            return true;
        } catch (...) {
            return true;
        }
        return false;
    }

    std::vector<std::string> split(const std::string &s, char delim)
    {
        std::vector<std::string> result;
        std::stringstream ss(s);
        std::string item;
        while (getline(ss, item, delim)) {
            result.push_back(item);
        }
        return result;
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
        if (decoder_) decoder_->Release();
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
    llm_tts() : StackFlow("melotts")
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

    bool is_breakpoint(const std::string &cutf8)
    {
        if (cutf8 == "，" || cutf8 == "、" || cutf8 == "," || cutf8 == "。" || cutf8 == "." || cutf8 == "!" ||
            cutf8 == "！" || cutf8 == "?" || cutf8 == "？" || cutf8 == ";" || cutf8 == "；")
            return true;
        else
            return false;
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
        std::vector<std::string> tmp_data = llm_task_obj->lexicon_->splitEachChar(user_msg);
        for (auto cutf8 : tmp_data) {
            if (is_breakpoint(cutf8)) {
                llm_task_obj->tts_string_stream_buff += cutf8;
                ret = llm_task_obj->TTS(llm_task_obj->tts_string_stream_buff, false);
                llm_task_obj->tts_string_stream_buff.clear();
                if (ret) {
                    error_body["code"]    = -11;
                    error_body["message"] = "Model run failed.";
                    llm_channel->send("None", "None", error_body, llm_channel->work_id_);
                }
            } else {
                llm_task_obj->tts_string_stream_buff += cutf8;
            }
        }
        if (finish_flage) {
            if (!llm_task_obj->tts_string_stream_buff.empty()) {
                llm_task_obj->tts_string_stream_buff.push_back('.');
                ret = llm_task_obj->TTS(llm_task_obj->tts_string_stream_buff, true);
                llm_task_obj->tts_string_stream_buff.clear();
                if (ret) {
                    error_body["code"]    = -11;
                    error_body["message"] = "Model run failed.";
                    llm_channel->send("None", "None", error_body, llm_channel->work_id_);
                }
            } else {
                llm_task_obj->TTS("", true);
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
            llm_task_obj->tts_string_stream_buff.clear();
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
            send("None", "None", error_body, "melotts");
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
            send("None", "None", error_body, "melotts");
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
            send("None", "None", error_body, "melotts");
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_melotts::link:%s", data.c_str());
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
        SLOGI("llm_melotts::unlink:%s", data.c_str());
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
        SLOGI("llm_melotts::taskinfo:%s", data.c_str());
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("melotts.tasklist", req_body, LLM_NO_ERROR, work_id);
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
            send("melotts.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_melotts::exit:%s", data.c_str());

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
