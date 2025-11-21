/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "runner/LLM.hpp"

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <stdexcept>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "thread_safe_list.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_vlm will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;

typedef struct {
    cv::Mat inference_src;
    bool inference_bgr2rgb;
} inference_async_par;

typedef std::function<void(const std::string &data, bool finish)> task_callback_t;

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

#define QWEN_CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))                \
        qwen_mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))                   \
        qwen_mode_config_.key = obj[#key];

class llm_task {
private:
    static std::atomic<unsigned int> next_port_;
    std::atomic_bool tokenizer_server_flage_;
    unsigned int port_;
    pid_t tokenizer_pid_ = -1;
    enum class ModelType { Unknown = 0, Qwen, InternVL, InternVL_CTX };
    ModelType model_type_ = ModelType::Unknown;

public:
    LLMAttrType mode_config_;
    Config qwen_mode_config_;
    std::unique_ptr<LLM> lLaMa_;
    std::unique_ptr<LLM_CTX> lLaMa_ctx_;
    std::unique_ptr<LLM_Qwen> qwen_;
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    std::vector<unsigned short> prompt_data_;
    std::vector<unsigned char> image_data_;
    std::vector<std::vector<unsigned char>> images_data;
    std::vector<cv::Mat> mats;
    std::vector<unsigned short> img_embed;
    std::vector<std::vector<unsigned short>> imgs_embed;
    std::vector<std::vector<float>> deepstack_features;
    std::vector<int> visual_pos_mask;
    std::vector<std::vector<int>> position_ids;
    std::string prompt_;
    std::string last_reply;
    std::vector<int> tokens_ids, tokens_diff;
    std::vector<std::vector<unsigned short>> k_caches, v_caches;
    std::string kvcache_path;
    int precompute_len = 0;
    std::vector<int> _token_ids;
    task_callback_t out_callback_;
    bool enoutput_;
    bool enstream_;
    bool encamera_;
    thread_safe::list<inference_async_par> async_list_;

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
            prompt_          = config_body.at("prompt");

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

    void vlm_reset()
    {
        std::vector<unsigned short>().swap(prompt_data_);

        for (auto &inner_vec : imgs_embed) {
            std::vector<unsigned short>().swap(inner_vec);
        }
        std::vector<std::vector<unsigned short>>().swap(imgs_embed);

        for (auto &inner_vec : deepstack_features) {
            std::vector<float>().swap(inner_vec);
        }
        std::vector<std::vector<float>>().swap(deepstack_features);

        std::vector<int>().swap(visual_pos_mask);

        for (auto &inner_vec : position_ids) {
            std::vector<int>().swap(inner_vec);
        }
        std::vector<std::vector<int>>().swap(position_ids);
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

            CONFIG_AUTO_SET(file_body["mode_param"], system_prompt);
            CONFIG_AUTO_SET(file_body["mode_param"], tokenizer_type);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_tokenizer_model);
            CONFIG_AUTO_SET(file_body["mode_param"], url_tokenizer_model);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_tokens_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_post_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_vpm_resampler_axmodedl);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_image_encoder_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], template_filename_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], b_vpm_two_stage);
            CONFIG_AUTO_SET(file_body["mode_param"], b_bos);
            CONFIG_AUTO_SET(file_body["mode_param"], b_eos);
            CONFIG_AUTO_SET(file_body["mode_param"], axmodel_num);
            CONFIG_AUTO_SET(file_body["mode_param"], tokens_embed_num);
            CONFIG_AUTO_SET(file_body["mode_param"], img_token_id);
            CONFIG_AUTO_SET(file_body["mode_param"], tokens_embed_size);
            CONFIG_AUTO_SET(file_body["mode_param"], b_use_mmap_load_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], b_dynamic_load_axmodel_layer);
            CONFIG_AUTO_SET(file_body["mode_param"], max_token_len);
            CONFIG_AUTO_SET(file_body["mode_param"], enable_temperature);
            CONFIG_AUTO_SET(file_body["mode_param"], temperature);
            CONFIG_AUTO_SET(file_body["mode_param"], enable_top_p_sampling);
            CONFIG_AUTO_SET(file_body["mode_param"], top_p);
            CONFIG_AUTO_SET(file_body["mode_param"], enable_top_k_sampling);
            CONFIG_AUTO_SET(file_body["mode_param"], top_k);
            CONFIG_AUTO_SET(file_body["mode_param"], enable_repetition_penalty);
            CONFIG_AUTO_SET(file_body["mode_param"], repetition_penalty);
            CONFIG_AUTO_SET(file_body["mode_param"], penalty_window);
            CONFIG_AUTO_SET(file_body["mode_param"], vpm_width);
            CONFIG_AUTO_SET(file_body["mode_param"], vpm_height);
            CONFIG_AUTO_SET(file_body["mode_param"], precompute_len);
            CONFIG_AUTO_SET(file_body["mode_param"], b_video);

            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_config.temporal_patch_size);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_config.tokens_per_second);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_config.spatial_merge_size);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_config.patch_size);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_config.width);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_config.height);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_config.fps);

            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], image_token_id);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], video_token_id);
            QWEN_CONFIG_AUTO_SET(file_body["mode_param"], vision_start_token_id);
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
                        setenv("PYTHONPATH", "/opt/m5stack/lib/vlm/site-packages", 1);
                        const std::string port_str = std::to_string(port_);
                        const std::string model_id = base_model + "tokenizer";

                        execl("/usr/bin/python3", "python3", tokenizer_file.c_str(), "--host", "localhost", "--port",
                              port_str.c_str(), "--model_id", model_id.c_str(), "--content", prompt_.c_str(),
                              (char *)nullptr);

                        perror("execl failed");
                        _exit(1);
                    }

                    tokenizer_server_flage_.store(true);
                    SLOGI("port_=%s model_id=%s content=%s", std::to_string(port_).c_str(),
                          (base_model + std::string("tokenizer")).c_str(), prompt_.c_str());

                    std::this_thread::sleep_for(std::chrono::seconds(15));
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

            {
                std::string encoder_name = mode_config_.filename_image_encoder_axmodel;
                std::transform(encoder_name.begin(), encoder_name.end(), encoder_name.begin(), ::tolower);

                if (encoder_name.find("qwen3") != std::string::npos)
                    model_type_ = ModelType::Qwen;
                else if (encoder_name.find("internvl3") != std::string::npos && mode_config_.precompute_len > 0)
                    model_type_ = ModelType::InternVL_CTX;
                else if (encoder_name.find("internvl3") != std::string::npos)
                    model_type_ = ModelType::InternVL;
                else
                    model_type_ = ModelType::Unknown;
            }
            mode_config_.filename_tokens_embed           = base_model + mode_config_.filename_tokens_embed;
            mode_config_.filename_post_axmodel           = base_model + mode_config_.filename_post_axmodel;
            mode_config_.filename_image_encoder_axmodel  = base_model + mode_config_.filename_image_encoder_axmodel;
            mode_config_.template_filename_axmodel       = base_model + mode_config_.template_filename_axmodel;
            mode_config_.filename_vpm_resampler_axmodedl = base_model + mode_config_.filename_vpm_resampler_axmodedl;
            mode_config_.runing_callback = [this](int *p_token, int n_token, const char *p_str, float token_per_sec,
                                                  void *reserve) {
                if (this->out_callback_) {
                    this->out_callback_(std::string(p_str), false);
                }
            };

            switch (model_type_) {
                case ModelType::InternVL: {
                    lLaMa_ = std::make_unique<LLM>();
                    if (!lLaMa_->Init(mode_config_)) {
                        lLaMa_->Deinit();
                        lLaMa_.reset();
                        return -2;
                    }
                    break;
                }

                case ModelType::InternVL_CTX: {
                    lLaMa_ctx_ = std::make_unique<LLM_CTX>();
                    if (!lLaMa_ctx_->Init(mode_config_)) {
                        lLaMa_ctx_->Deinit();
                        lLaMa_ctx_.reset();
                        return -2;
                    }
                    break;
                }

                case ModelType::Qwen: {
                    qwen_ = std::make_unique<LLM_Qwen>();
                    if (!qwen_->Init(mode_config_)) {
                        qwen_->Deinit();
                        qwen_.reset();
                        return -2;
                    }
                    break;
                }
                default:
                    ALOGE("Unknown model type in filename_image_encoder_axmodel: %s",
                          mode_config_.filename_image_encoder_axmodel.c_str());
                    return -3;
            }

            if (lLaMa_ctx_) {
                lLaMa_ctx_->SetSystemPrompt(mode_config_.system_prompt, _token_ids);
                if (!kvcache_path.empty()) {
                    prepare_kvcache_folder(kvcache_path);
                }
                if (!kvcache_path.empty() && kvcache_path != "") {
                    if (lLaMa_ctx_->load_kvcache(kvcache_path, mode_config_.axmodel_num, k_caches, v_caches,
                                                 mode_config_.system_prompt, precompute_len)) {
                        ALOGI("load kvcache from path: %s success,precompute_len: %d", kvcache_path.c_str(),
                              precompute_len);
                    } else {
                        ALOGW("load kvcache from path: %s failed,generate kvcache", kvcache_path.c_str());
                        lLaMa_ctx_->GenerateKVCachePrefill(_token_ids, k_caches, v_caches, precompute_len);
                        if (!lLaMa_ctx_->save_kvcache(kvcache_path, mode_config_.system_prompt, precompute_len,
                                                      k_caches, v_caches)) {
                            ALOGE("save kvcache failed");
                        }
                        ALOGI("generate kvcache to path: %s", kvcache_path.c_str());
                    }
                } else {
                    lLaMa_ctx_->GenerateKVCachePrefill(_token_ids, k_caches, v_caches, precompute_len);
                }
                ALOGI("precompute_len: %d", precompute_len);
                ALOGI("system_prompt: %s", mode_config_.system_prompt.c_str());
            }
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
            case TKT_HTTP:
            default:
                oss_prompt << input;
                break;
        }
        // SLOGI("prompt_complete:%s", oss_prompt.str().c_str());
        return oss_prompt.str();
    }

    int inference_async(cv::Mat &src, bool bgr2rgb = true)
    {
        if (async_list_.size() < 1) {
            inference_async_par par;
            par.inference_src     = src.clone();
            par.inference_bgr2rgb = bgr2rgb;
            async_list_.put(par);
        }
        return async_list_.size();
    }

    bool inference_raw_yuv(const std::string &msg)
    {
        if (msg.size() != 320 * 320 * 2) {
            throw std::string("img size error");
        }
        cv::Mat camera_data(320, 320, CV_8UC2, (void *)msg.data());
        cv::Mat rgb;
        cv::cvtColor(camera_data, rgb, cv::COLOR_YUV2RGB_YUYV);
        return inference_async(rgb, true) ? false : true;
    }

    void inference(const std::string &msg)
    {
        try {
            if (lLaMa_) {
                if (encamera_) {
                    inference_async_par par;
                    async_list_.get();  // discard buffered frames
                    par = async_list_.get();
                    if (par.inference_src.empty()) return;
                    if (par.inference_bgr2rgb) {
                        cv::Mat rgb;
                        cv::cvtColor(par.inference_src, rgb, cv::COLOR_BGR2RGB);
                        par.inference_src = rgb;
                    }
                    lLaMa_->Encode(par.inference_src, img_embed);
                    lLaMa_->Encode(img_embed, prompt_data_, prompt_complete(msg));
                    std::string out = lLaMa_->Run(prompt_data_);
                    if (out_callback_) out_callback_(out, true);
                } else if (image_data_.empty()) {
                    lLaMa_->Encode(prompt_data_, prompt_complete(msg));
                    std::string out = lLaMa_->Run(prompt_data_);
                    if (out_callback_) out_callback_(out, true);
                } else {
                    cv::Mat src = cv::imdecode(image_data_, cv::IMREAD_COLOR);
                    if (src.empty()) return;
                    image_data_.clear();
                    lLaMa_->Encode(src, img_embed);
                    lLaMa_->Encode(img_embed, prompt_data_, prompt_complete(msg));
                    std::string out = lLaMa_->Run(prompt_data_);
                    if (out_callback_) out_callback_(out, true);
                }
            }

            if (lLaMa_ctx_) {
                if (msg == "reset") {
                    lLaMa_ctx_->SetSystemPrompt(mode_config_.system_prompt, _token_ids);
                    lLaMa_ctx_->GenerateKVCachePrefill(_token_ids, k_caches, v_caches, precompute_len);
                    last_reply.clear();
                    mats.clear();
                    if (out_callback_) out_callback_("Context has been reset.", true);
                    return;
                }

                if (images_data.empty()) {
                    lLaMa_ctx_->Encode(prompt_data_, prompt_complete(msg), last_reply, tokens_ids, tokens_diff);
                    if (auto ret = lLaMa_ctx_->SetKVCache(k_caches, v_caches, precompute_len, tokens_diff.size());
                        ret != 0) {
                        ALOGW("The context full,Reset context");
                        lLaMa_ctx_->SetSystemPrompt(mode_config_.system_prompt, _token_ids);
                        lLaMa_ctx_->GenerateKVCachePrefill(_token_ids, k_caches, v_caches, precompute_len);
                        lLaMa_ctx_->SetKVCache(k_caches, v_caches, precompute_len, tokens_diff.size());
                    }
                    last_reply = lLaMa_ctx_->Run(prompt_data_);
                    lLaMa_ctx_->GetKVCache(k_caches, v_caches, precompute_len);
                    if (out_callback_) out_callback_(last_reply, true);
                } else {
                    for (const auto &img_buf : images_data) {
                        cv::Mat src = cv::imdecode(img_buf, cv::IMREAD_COLOR);
                        if (src.empty()) {
                            std::cerr << "Decode failed!" << std::endl;
                            continue;
                        }
                        mats.push_back(src);
                    }
                    if (mats.empty()) return;
                    images_data.clear();
                    lLaMa_ctx_->ClearImgsEmbed();
                    std::vector<std::vector<unsigned short>> all_embeds;
                    if (auto ret = lLaMa_ctx_->Encode(mats, all_embeds); ret != 0) {
                        ALOGE("lLaMaCtx.Encode failed");
                        if (out_callback_) out_callback_("Encode failed", true);
                        return;
                    }
                    mats.clear();
                    if (auto ret =
                            lLaMa_ctx_->Encode(all_embeds, prompt_data_, prompt_complete(msg), tokens_ids, tokens_diff);
                        ret != 0) {
                        ALOGE("lLaMaCtx.Encode failed");
                        if (out_callback_) out_callback_("Encode failed", true);
                        return;
                    }
                    if (auto ret = lLaMa_ctx_->SetKVCache(k_caches, v_caches, precompute_len, tokens_diff.size());
                        ret != 0) {
                        ALOGW("The context full,Reset context");
                        lLaMa_ctx_->SetSystemPrompt(mode_config_.system_prompt, _token_ids);
                        lLaMa_ctx_->GenerateKVCachePrefill(_token_ids, k_caches, v_caches, precompute_len);
                        lLaMa_ctx_->SetKVCache(k_caches, v_caches, precompute_len, tokens_diff.size());
                        lLaMa_ctx_->ClearImgsEmbed();
                    }
                    last_reply = lLaMa_ctx_->Run(prompt_data_);
                    lLaMa_ctx_->GetKVCache(k_caches, v_caches, precompute_len);
                    if (out_callback_) out_callback_(last_reply, true);
                }
            }

            if (qwen_) {
                if (images_data.empty()) {
                    qwen_->Encode(prompt_data_, position_ids, qwen_mode_config_, prompt_complete(msg));
                    last_reply = qwen_->Run(prompt_data_, position_ids, deepstack_features, visual_pos_mask);
                    if (out_callback_) out_callback_(last_reply, true);
                } else {
                    for (const auto &img_buf : images_data) {
                        cv::Mat src = cv::imdecode(img_buf, cv::IMREAD_COLOR);
                        if (src.empty()) {
                            std::cerr << "Decode failed!" << std::endl;
                            continue;
                        }
                        mats.push_back(src);
                    }
                    images_data.clear();
                    if (mats.empty()) return;
                    std::vector<std::vector<unsigned short>> all_embeds;
                    qwen_->EncodeImage(mats, mode_config_.b_video, qwen_mode_config_, all_embeds, deepstack_features);
                    mats.clear();
                    qwen_->Encode(all_embeds, mode_config_.b_video, prompt_data_, position_ids, visual_pos_mask,
                                  qwen_mode_config_, prompt_complete(msg));
                    last_reply = qwen_->Run(prompt_data_, position_ids, deepstack_features, visual_pos_mask);
                    if (out_callback_) out_callback_(last_reply, true);
                    vlm_reset();
                }
            }
        } catch (...) {
            SLOGW("lLaMa_->Run have error!");
        }
    }

    bool pause()
    {
        if (lLaMa_) lLaMa_->Stop();
        if (lLaMa_ctx_) lLaMa_ctx_->Stop();
        if (qwen_) qwen_->Stop();
        return true;
    }

    bool delete_model()
    {
        if (tokenizer_pid_ != -1) {
            kill(tokenizer_pid_, SIGTERM);
            waitpid(tokenizer_pid_, nullptr, 0);
            tokenizer_pid_ = -1;
        }
        if (lLaMa_) lLaMa_->Deinit();
        if (lLaMa_) lLaMa_.reset();
        if (qwen_) qwen_->Deinit();
        if (qwen_) qwen_.reset();
        return true;
    }

    static unsigned int getNextPort()
    {
        unsigned int port = next_port_++;
        if (port > 8099) {
            next_port_ = 8090;
            port       = 8090;
        }
        return port;
    }

    llm_task(const std::string &workid) : tokenizer_server_flage_(false), port_(getNextPort())
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
        if (tokenizer_pid_ != -1) {
            kill(tokenizer_pid_, SIGTERM);
            waitpid(tokenizer_pid_, nullptr, WNOHANG);
        }
        if (lLaMa_) {
            lLaMa_->Deinit();
        }
        if (lLaMa_ctx_) {
            lLaMa_ctx_->Deinit();
        }
        if (qwen_) {
            qwen_->Deinit();
        }
    }
};

std::atomic<unsigned int> llm_task::next_port_{8090};

#undef CONFIG_AUTO_SET

class llm_vlm : public StackFlow {
private:
    int task_count_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_vlm() : StackFlow("vlm")
    {
        task_count_ = 2;
    }

    void task_output(const std::weak_ptr<llm_task> llm_task_obj_weak,
                     const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &data, bool finish)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        SLOGI("send:%s", data.c_str());
        if (llm_channel->enstream_) {
            static int count = 0;
            nlohmann::json data_body;
            data_body["index"] = count++;
            data_body["delta"] = data;
            if (!finish)
                data_body["delta"] = data;
            else
                data_body["delta"] = std::string("");
            data_body["finish"] = finish;
            if (finish) count = 0;
            SLOGI("send stream");
            llm_channel->send(llm_task_obj->response_format_, data_body, LLM_NO_ERROR);
        } else if (finish) {
            SLOGI("send utf-8");
            llm_channel->send(llm_task_obj->response_format_, data, LLM_NO_ERROR);
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
        if (llm_task_obj->lLaMa_) llm_task_obj->lLaMa_->Stop();
        if (llm_task_obj->lLaMa_ctx_) llm_task_obj->lLaMa_ctx_->Stop();
        if (llm_task_obj->qwen_) llm_task_obj->qwen_->Stop();
    }

    void pause(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_vlm::work:%s", data.c_str());

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
        if (object.find("jpeg") != std::string::npos) {
            llm_task_obj->image_data_.assign(next_data->begin(), next_data->end());
            llm_task_obj->images_data.emplace_back(next_data->begin(), next_data->end());
            return;
        }
        llm_task_obj->inference((*next_data));
    }

    void task_asr_data(const std::weak_ptr<llm_task> llm_task_obj_weak,
                       const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                       const std::string &data)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        if (object.find("stream") != std::string::npos) {
            if (sample_json_str_get(data, "finish") == "true") {
                llm_task_obj->inference(sample_json_str_get(data, "delta"));
            }
        } else {
            llm_task_obj->inference(data);
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
        if (llm_task_obj->lLaMa_) llm_task_obj->lLaMa_->Stop();
        if (llm_task_obj->lLaMa_ctx_) llm_task_obj->lLaMa_ctx_->Stop();
        if (llm_task_obj->qwen_) llm_task_obj->qwen_->Stop();
    }

    void task_camera_data(const std::weak_ptr<llm_task> llm_task_obj_weak,
                          const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &data)
    {
        nlohmann::json error_body;
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            SLOGE("Model run failed.");
            return;
        }
        try {
            llm_task_obj->inference_raw_yuv(data);
        } catch (...) {
            SLOGE("data format error");
        }
    }

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        if ((llm_task_channel_.size() - 1) == task_count_) {
            error_body["code"]    = -21;
            error_body["message"] = "task full";
            send("None", "None", error_body, "vlm");
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

            llm_task_obj->set_output(std::bind(&llm_vlm::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("vlm") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_vlm::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if (input.find("asr") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_vlm::task_asr_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                } else if (input.find("kws") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_vlm::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                } else if (input.find("camera") != std::string::npos) {
                    llm_task_obj->encamera_    = true;
                    std::string input_url_name = input + ".out_port";
                    std::string input_url      = unit_call("sys", "sql_select", input_url_name);
                    if (!input_url.empty()) {
                        std::weak_ptr<llm_task> _llm_task_obj       = llm_task_obj;
                        std::weak_ptr<llm_channel_obj> _llm_channel = llm_channel;
                        llm_channel->subscriber(input_url, [this, _llm_task_obj, _llm_channel](
                                                               pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                            this->task_camera_data(_llm_task_obj, _llm_channel, raw->string());
                        });
                    }
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
            send("None", "None", error_body, "vlm");
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_vlm::link:%s", data.c_str());
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
        if (data.find("asr") != std::string::npos) {
            ret = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_vlm::task_asr_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("kws") != std::string::npos) {
            ret = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_vlm::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("camera") != std::string::npos) {
            llm_task_obj->encamera_    = true;
            std::string input_url_name = data + ".out_port";
            std::string input_url      = unit_call("sys", "sql_select", input_url_name);
            if (!input_url.empty()) {
                std::weak_ptr<llm_task> _llm_task_obj       = llm_task_obj;
                std::weak_ptr<llm_channel_obj> _llm_channel = llm_channel;
                llm_channel->subscriber(
                    input_url, [this, _llm_task_obj, _llm_channel](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                        this->task_camera_data(_llm_task_obj, _llm_channel, raw->string());
                    });
                ret = 0;
            }
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
        SLOGI("llm_vlm::unlink:%s", data.c_str());
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
        SLOGI("llm_vlm::taskinfo:%s", data.c_str());
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("vlm.tasklist", req_body, LLM_NO_ERROR, work_id);
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
            send("vlm.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_vlm::exit:%s", data.c_str());

        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return -1;
        }
        llm_task_[work_id_num]->stop();
        task_pause(llm_task_[work_id_num], get_channel(work_id_num));
        auto llm_channel = get_channel(work_id_num);
        llm_channel->stop_subscriber("");
        llm_task_.erase(work_id_num);
        send("None", "None", LLM_NO_ERROR, work_id);
        return 0;
    }

    ~llm_vlm()
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
    llm_vlm llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}