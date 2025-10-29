/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "runner/LLM.hpp"
#include "runner/Token2wav.hpp"

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <stdexcept>
#include <samplerate.h>
#include <semaphore.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "thread_safe_list.h"
using namespace StackFlows;
#ifdef ENABLE_BACKWARD
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
#include "backward.h"
#endif

#include "runner/utils/wav.hpp"

#define MAX_TASK_NUM 2

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_cosy_voice will be exit!");
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
    static std::atomic<unsigned int> next_port_;
    std::atomic_bool tokenizer_server_flage_;
    unsigned int port_;
    pid_t tokenizer_pid_ = -1;
    std::mutex g_buffer_mutex;
    std::condition_variable g_buffer_cv;
    std::atomic<bool> g_llm_finished{false};
    std::atomic<bool> g_stop{false};
    TokenBuffer g_token_buffer;
    Token2Wav lToken2Wav;

    std::vector<int> prompt_text_token;
    std::vector<unsigned short> prompt_text_embeds;
    std::vector<int> prompt_speech_token;
    std::vector<unsigned short> prompt_speech_embeds;

    std::vector<float> prompt_feat;
    std::vector<float> prompt_speech_embeds_flow;
    std::vector<float> spk_embeds;

public:
    enum inference_status { INFERENCE_NONE = 0, INFERENCE_RUNNING };
    LLMAttrType mode_config_;
    std::unique_ptr<LLM> lLaMa_;
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    std::string prompt_;
    std::string last_reply;
    std::vector<unsigned short> prompt_data;
    std::vector<int> tokens_ids, tokens_diff;
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    std::string kvcache_path;
    int precompute_len = 0;
    std::vector<int> _token_ids;
    static int ax_init_flage_;
    task_callback_t out_callback_;
    bool enoutput_;
    bool enstream_;

    std::unique_ptr<std::thread> inference_run_;
    thread_safe::list<std::string> async_list_;

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

    void prepare_kvcache_folder(const std::string &kvcache_path)
    {
        try {
            if (!std::filesystem::exists(kvcache_path)) {
                std::filesystem::create_directories(kvcache_path);
            }

            if (std::filesystem::exists(kvcache_path) && std::filesystem::is_directory(kvcache_path)) {
                for (const auto &entry : std::filesystem::directory_iterator(kvcache_path)) {
                    std::filesystem::remove_all(entry.path());
                }
            }
        } catch (const std::exception &e) {
            ALOGI("prepare_kvcache_folder: skip clear/create due to error: %s", e.what());
        }
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
            CONFIG_AUTO_SET(file_body["mode_param"], mode_rate);
            CONFIG_AUTO_SET(file_body["mode_param"], audio_rate);
            CONFIG_AUTO_SET(file_body["mode_param"], tokenizer_type);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_tokenizer_model);
            CONFIG_AUTO_SET(file_body["mode_param"], url_tokenizer_model);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_post_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_tokens_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_llm_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_speech_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_decoder_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], token2wav_axmodel_dir);
            CONFIG_AUTO_SET(file_body["mode_param"], n_timesteps);
            CONFIG_AUTO_SET(file_body["mode_param"], prompt_files);
            CONFIG_AUTO_SET(file_body["mode_param"], template_filename_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], b_bos);
            CONFIG_AUTO_SET(file_body["mode_param"], b_eos);
            CONFIG_AUTO_SET(file_body["mode_param"], axmodel_num);
            CONFIG_AUTO_SET(file_body["mode_param"], tokens_embed_num);
            CONFIG_AUTO_SET(file_body["mode_param"], tokens_embed_size);
            CONFIG_AUTO_SET(file_body["mode_param"], b_use_mmap_load_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], b_dynamic_load_axmodel_layer);
            CONFIG_AUTO_SET(file_body["mode_param"], max_token_len);
            {
                auto has_http = [](const std::string &s) { return s.find("http") != std::string::npos; };

                auto find_tokenizer_file = [this]() -> std::string {
                    const std::string base = "/opt/m5stack/scripts/";
                    const std::string a    = base + model_ + "_tokenizer.py";
                    if (file_exists(a)) return a;
                    const std::string b = base + "tokenizer_" + model_ + ".py";
                    if (file_exists(b)) return b;
                    SLOGE("%s or %s not found!", a.c_str(), b.c_str());
                    return {};
                };

                auto start_tokenizer_server = [&](const std::string &tokenizer_file) {
                    if (tokenizer_file.empty()) return;
                    if (tokenizer_server_flage_.load()) return;
                    tokenizer_pid_ = fork();
                    if (tokenizer_pid_ == 0) {
                        FILE *fp  = popen("python3 -V 2>&1", "r");
                        int major = 3, minor = 11;
                        if (fp) {
                            char buf[64] = {0};
                            if (fgets(buf, sizeof(buf), fp)) {
                                sscanf(buf, "Python %d.%d", &major, &minor);
                            }
                            pclose(fp);
                        }

                        std::string python_path = "/opt/m5stack/lib/cosy-voice/python" + std::to_string(major) + "." +
                                                  std::to_string(minor) + "/site-packages";

                        setenv("PYTHONPATH", python_path.c_str(), 1);

                        const std::string port_str = std::to_string(port_);
                        const std::string model_id = base_model + "tokenizer";

                        execl("/usr/bin/python3", "python3", tokenizer_file.c_str(), "--host", "localhost", "--port",
                              port_str.c_str(), "--model_id", model_id.c_str(), (char *)nullptr);

                        perror("execl failed");
                        _exit(1);
                    }

                    tokenizer_server_flage_.store(true);
                    SLOGI("port_=%s model_id=%s content=%s", std::to_string(port_).c_str(),
                          (base_model + std::string("tokenizer")).c_str(), prompt_.c_str());

                    std::this_thread::sleep_for(std::chrono::seconds(3));
                };

                auto process_field = [&](std::string &field, const char *name_for_log) -> bool {
                    if (!has_http(field)) return false;

                    field                            = "http://localhost:" + std::to_string(port_);
                    const std::string tokenizer_file = find_tokenizer_file();
                    start_tokenizer_server(tokenizer_file);
                    SLOGI("%s: %s", name_for_log, field.c_str());
                    return true;
                };

                if (!process_field(mode_config_.filename_tokenizer_model, "filename_tokenizer_model") &&
                    !process_field(mode_config_.url_tokenizer_model, "url_tokenizer_model")) {
                    mode_config_.filename_tokenizer_model = base_model + mode_config_.filename_tokenizer_model;
                    SLOGI("filename_tokenizer_model: %s", mode_config_.filename_tokenizer_model.c_str());
                }
            }
            mode_config_.filename_tokens_embed     = base_model + mode_config_.filename_tokens_embed;
            mode_config_.filename_post_axmodel     = base_model + mode_config_.filename_post_axmodel;
            mode_config_.filename_llm_embed        = base_model + mode_config_.filename_llm_embed;
            mode_config_.template_filename_axmodel = base_model + mode_config_.template_filename_axmodel;
            mode_config_.filename_speech_embed     = base_model + mode_config_.filename_speech_embed;
            mode_config_.filename_decoder_axmodel  = base_model + mode_config_.filename_decoder_axmodel;
            mode_config_.token2wav_axmodel_dir     = base_model + mode_config_.token2wav_axmodel_dir;
            mode_config_.prompt_files              = base_model + mode_config_.prompt_files;
            mode_config_.runing_callback = [this](int *p_token, int n_token, const char *p_str, float token_per_sec,
                                                  void *reserve) {
                if (this->out_callback_) {
                    this->out_callback_(std::string(p_str), false);
                }
            };

            readtxt(mode_config_.prompt_files + "/prompt_text.txt", prompt_text_token);
            readtxt(mode_config_.prompt_files + "/llm_prompt_speech_token.txt", prompt_speech_token);
            readtxt(mode_config_.prompt_files + "/prompt_speech_feat.txt", prompt_feat);
            readtxt<float>(mode_config_.prompt_files + "/flow_embedding.txt", spk_embeds);

            lLaMa_ = std::make_unique<LLM>();
            if (!lLaMa_->Init(mode_config_)) {
                lLaMa_->Deinit();
                lLaMa_.reset();
                return -2;
            }
            if (!lToken2Wav.Init(mode_config_.token2wav_axmodel_dir, mode_config_.n_timesteps)) {
                return -1;
            }
            lLaMa_->TextToken2Embeds(prompt_text_token, prompt_text_embeds);
            lLaMa_->SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds);
            lToken2Wav.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds_flow);

        } catch (...) {
            SLOGE("config false");
            return -3;
        }
        return 0;
    }

    std::string prompt_complete(const std::string &input)
    {
        std::ostringstream oss_prompt;
        switch (mode_config_.tokenizer_type) {
            case TKT_LLaMa:
                oss_prompt << "<|user|>\n" << input << "</s><|assistant|>\n";
                break;
            case TKT_Phi3:
                oss_prompt << input << " ";
                break;
            case TKT_Qwen:
                oss_prompt << "<|im_start|>system\n" << prompt_ << ".<|im_end|>";
                oss_prompt << "\n<|im_start|>user\n" << input << "<|im_end|>\n<|im_start|>assistant\n";
                break;
            case TKT_HTTP:
            default:
                oss_prompt << input;
                break;
        }
        SLOGI("prompt_complete:%s", oss_prompt.str().c_str());
        return oss_prompt.str();
    }

    void reset()
    {
        g_llm_finished = false;
        g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
        lToken2Wav.reset();
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

    int tts(const std::string &text, std::vector<unsigned short> &prompt_text_embeds,
            std::vector<unsigned short> &prompt_speech_embeds, const std::vector<float> &prompt_feat,
            const std::vector<float> &prompt_speech_embeds_flow, std::vector<float> &spk_embeds)
    {
        std::vector<float> output;
        timer time_total;
        time_total.start();
        try {
            auto llm_thread_func = [this, &text, &prompt_text_embeds, &prompt_speech_embeds]() {
                lLaMa_->Run(text, prompt_text_embeds, prompt_speech_embeds, g_token_buffer, g_buffer_mutex, g_buffer_cv,
                            g_llm_finished);
            };

            std::thread llm_thread(llm_thread_func);

            int token_offset     = 0;
            int prompt_token_len = prompt_speech_embeds_flow.size() / lToken2Wav.flow_embed_size;
            if (prompt_token_len < 75) {
                SLOGE("Error, prompt speech token len %d < 75", prompt_token_len);
                return -1;
            }
            int prompt_token_align_len = 75;

            std::vector<float> prompt_speech_embeds_flow1;
            prompt_speech_embeds_flow1.insert(prompt_speech_embeds_flow1.begin(), prompt_speech_embeds_flow.begin(),
                                              prompt_speech_embeds_flow.begin() + prompt_token_align_len * 512);

            std::vector<float> prompt_feat1;
            prompt_feat1.insert(prompt_feat1.begin(), prompt_feat.begin(),
                                prompt_feat.begin() + prompt_token_align_len * 2 * 80);

            int promot_token_pad = 0;
            int this_token_hop_len;
            int i = 0;
            while (true) {
                this_token_hop_len =
                    (token_offset == 0) ? lToken2Wav.token_hop_len + promot_token_pad : lToken2Wav.token_hop_len;

                std::unique_lock<std::mutex> lock(g_buffer_mutex);

                g_buffer_cv.wait(lock, [&] {
                    return (g_token_buffer.size() - token_offset >=
                            this_token_hop_len + lToken2Wav.pre_lookahead_len) ||
                           g_llm_finished.load() || g_stop.load();
                });

                if (g_stop) {
                    lock.unlock();
                    break;
                } else if (g_token_buffer.size() - token_offset >= this_token_hop_len + lToken2Wav.pre_lookahead_len) {
                    std::vector<SpeechToken> token;
                    int start = token_offset - std::min(int(token_offset / lToken2Wav.token_hop_len),
                                                        lToken2Wav.max_infer_chunk_num - 1) *
                                                   lToken2Wav.token_hop_len;
                    int end = token_offset + this_token_hop_len + lToken2Wav.pre_lookahead_len;

                    token.insert(token.end(), g_token_buffer.begin() + start, g_token_buffer.begin() + end);

                    lock.unlock();
                    auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds,
                                                   token_offset, false);
                    token_offset += this_token_hop_len;
                    output.insert(output.end(), speech.begin(), speech.end());
                    double src_ratio =
                        static_cast<double>(mode_config_.audio_rate) / static_cast<double>(mode_config_.mode_rate);
                    std::vector<float> resampled_pcm(static_cast<size_t>(speech.size() * src_ratio + 1));
                    int resampled_len = 0;
                    resample_audio(speech.data(), speech.size(), resampled_pcm.data(), &resampled_len, src_ratio);
                    std::vector<int16_t> wav_pcm_data;
                    wav_pcm_data.reserve(resampled_len);
                    for (int i = 0; i < resampled_len; i++) {
                        float val = resampled_pcm[i];
                        if (val > 1.0f) val = 1.0f;
                        if (val < -1.0f) val = -1.0f;
                        wav_pcm_data.push_back(static_cast<int16_t>(val * 32767.0f));
                    }

                    if (out_callback_) {
                        out_callback_(std::string(reinterpret_cast<char *>(wav_pcm_data.data()),
                                                  wav_pcm_data.size() * sizeof(int16_t)),
                                      false);
                    }
                    ++i;
                } else if (g_llm_finished.load()) {
                    lock.unlock();
                    break;
                } else {
                    lock.unlock();
                }
            }

            if (llm_thread.joinable()) {
                llm_thread.join();
            }

            if (g_stop) {
                g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
                return 1;
            }

            std::vector<SpeechToken> token;
            int start = g_token_buffer.size() - std::min(int(g_token_buffer.size() / lToken2Wav.token_hop_len),
                                                         lToken2Wav.max_infer_chunk_num - 1) *
                                                    lToken2Wav.token_hop_len;
            token.insert(token.end(), g_token_buffer.begin() + start, g_token_buffer.end());
            auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds,
                                           token_offset - start, true);
            double src_ratio =
                static_cast<double>(mode_config_.audio_rate) / static_cast<double>(mode_config_.mode_rate);
            std::vector<float> resampled_pcm(static_cast<size_t>(speech.size() * src_ratio + 1));
            int resampled_len = 0;
            resample_audio(speech.data(), speech.size(), resampled_pcm.data(), &resampled_len, src_ratio);

            std::vector<int16_t> wav_pcm_data;
            wav_pcm_data.reserve(resampled_len);
            for (int i = 0; i < resampled_len; i++) {
                float val = resampled_pcm[i];
                if (val > 1.0f) val = 1.0f;
                if (val < -1.0f) val = -1.0f;
                wav_pcm_data.push_back(static_cast<int16_t>(val * 32767.0f));
            }
            if (out_callback_) {
                out_callback_(
                    std::string(reinterpret_cast<char *>(wav_pcm_data.data()), wav_pcm_data.size() * sizeof(int16_t)),
                    true);
            }

            SLOGI("tts total use time: %.3f s", time_total.cost() / 1000);
            reset();
        } catch (const std::exception &e) {
            std::cerr << "Error in pipeline: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }

    void run()
    {
        std::string par;
        for (;;) {
            {
                par = async_list_.get();
                if (par.empty()) break;
                inference(par);
            }
        }
    }

    int inference_async(const std::string &msg)
    {
        if (msg.empty()) return -1;
        if (async_list_.size() < 3) {
            std::string par = msg;
            async_list_.put(par);
        } else {
            SLOGE("inference list is full\n");
        }
        return async_list_.size();
    }

    void inference(const std::string &msg)
    {
        try {
            // std::string out = lLaMa_->Run(prompt_complete(msg));
            // if (out_callback_) out_callback_(out, true);
            tts(msg, prompt_text_embeds, prompt_speech_embeds, prompt_feat, prompt_speech_embeds_flow, spk_embeds);
            std::string out = "finish";
            if (out_callback_) out_callback_(out, true);

        } catch (...) {
            SLOGW("lLaMa_->Run have error!");
        }
    }

    void _ax_init()
    {
        if (!ax_init_flage_) {
            int ret = axclInit(nullptr);
            if (0 != ret) {
                fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
            }
        }
        ax_init_flage_++;
    }

    void _ax_deinit()
    {
        if (ax_init_flage_ > 0) {
            --ax_init_flage_;
            if (!ax_init_flage_) {
                axclFinalize();
            }
        }
    }

    bool pause()
    {
        if (lLaMa_) lLaMa_->Stop();
        return true;
    }

    bool delete_model()
    {
        if (tokenizer_pid_ != -1) {
            kill(tokenizer_pid_, SIGTERM);
            waitpid(tokenizer_pid_, nullptr, 0);
            tokenizer_pid_ = -1;
        }
        lLaMa_->Deinit();
        lLaMa_.reset();
        return true;
    }

    static unsigned int getNextPort()
    {
        unsigned int port = next_port_++;
        if (port > 8079) {
            next_port_ = 8070;
            port       = 8070;
        }
        return port;
    }

    llm_task(const std::string &workid) : tokenizer_server_flage_(false), port_(getNextPort())
    {
        _ax_init();
        inference_run_ = std::make_unique<std::thread>(std::bind(&llm_task::run, this));
    }

    void start()
    {
        if (!inference_run_) {
            inference_run_ = std::make_unique<std::thread>(std::bind(&llm_task::run, this));
        }
    }

    void stop()
    {
        if (inference_run_) {
            std::string par;
            async_list_.put(par);
            if (lLaMa_) lLaMa_->Stop();
            inference_run_->join();
            inference_run_.reset();
        }
    }

    ~llm_task()
    {
        stop();
        if (tokenizer_pid_ != -1) {
            kill(tokenizer_pid_, SIGTERM);
            waitpid(tokenizer_pid_, nullptr, WNOHANG);
        }
        if (lLaMa_) {
            lLaMa_->Deinit();
        }
        _ax_deinit();
    }
};
int llm_task::ax_init_flage_ = 0;

std::atomic<unsigned int> llm_task::next_port_{8070};

#undef CONFIG_AUTO_SET

class llm_cosy_voice : public StackFlow {
private:
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_cosy_voice() : StackFlow("cosy_voice")
    {
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

    void task_pause(const std::weak_ptr<llm_task> llm_task_obj_weak,
                    const std::weak_ptr<llm_channel_obj> llm_channel_weak)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        llm_task_obj->lLaMa_->Stop();
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
        const std::string *next_data = &data;
        int ret;
        std::string tmp_msg1;
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
        llm_task_obj->inference_async(sample_unescapeString(*next_data));
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
        llm_task_obj->lLaMa_->Stop();
    }

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        if ((llm_task_channel_.size() - 1) == MAX_TASK_NUM) {
            error_body["code"]    = -21;
            error_body["message"] = "task full";
            send("None", "None", error_body, "llm");
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

            llm_task_obj->set_output(
                std::bind(&llm_cosy_voice::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("tts") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_cosy_voice::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if ((input.find("llm") != std::string::npos) || (input.find("vlm") != std::string::npos)) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_cosy_voice::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                } else if (input.find("kws") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_cosy_voice::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
            send("None", "None", error_body, "llm");
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_llm::link:%s", data.c_str());
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
        if (data.find("llm") != std::string::npos) {
            ret = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_cosy_voice::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("kws") != std::string::npos) {
            ret = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_cosy_voice::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
        SLOGI("llm_llm::unlink:%s", data.c_str());
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
        SLOGI("llm_llm::taskinfo:%s", data.c_str());
        // int ret = 0;
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("llm.tasklist", req_body, LLM_NO_ERROR, work_id);
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
            send("llm.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_llm::exit:%s", data.c_str());

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

    ~llm_cosy_voice()
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
    llm_cosy_voice llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}