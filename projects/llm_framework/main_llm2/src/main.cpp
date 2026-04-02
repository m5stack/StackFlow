/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "runner/LLM.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <base64.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <semaphore.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "thread_safe_list.h"
using namespace StackFlows;
#ifdef ENABLE_BACKWARD
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
#include "backward.h"
#endif

#define MAX_TASK_NUM 2

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_llm2 will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;

typedef std::function<void(const std::string &data, bool finish)> task_callback_t;

typedef struct {
    std::string prompt;
    std::vector<std::string> image_paths;
    std::vector<std::string> temp_files;
} inference_async_par;

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
    thread_safe::list<inference_async_par> async_list_;
    std::mutex pending_media_mutex_;
    std::vector<std::string> pending_image_paths_;
    std::vector<std::string> pending_temp_files_;

    static std::string save_image_to_tempfile(const std::string &image_bytes)
    {
        if (image_bytes.empty()) {
            return {};
        }

        std::filesystem::path tmpdir;
        try {
            tmpdir = std::filesystem::temp_directory_path() / "stackflow_llm2_images";
        } catch (...) {
            tmpdir = std::filesystem::current_path() / "tmp" / "stackflow_llm2_images";
        }

        std::error_code ec;
        std::filesystem::create_directories(tmpdir, ec);
        if (ec) {
            SLOGE("create temp dir failed: %s", ec.message().c_str());
            return {};
        }

        const auto now  = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto path = tmpdir / ("img_" + std::to_string(now) + "_" + std::to_string(getpid()) + ".jpg");

        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            SLOGE("open temp image failed: %s", path.string().c_str());
            return {};
        }

        ofs.write(image_bytes.data(), static_cast<std::streamsize>(image_bytes.size()));
        ofs.close();
        return path.string();
    }

    static void cleanup_temp_files(const std::vector<std::string> &files)
    {
        for (const auto &file : files) {
            std::error_code ec;
            std::filesystem::remove(file, ec);
        }
    }

    void cleanup_pending_media()
    {
        std::vector<std::string> temp_files;
        {
            std::lock_guard<std::mutex> lock(pending_media_mutex_);
            pending_image_paths_.clear();
            temp_files.swap(pending_temp_files_);
        }
        cleanup_temp_files(temp_files);
    }

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

            if (config_body.contains("prompt")) {
                mode_config_.system_prompt = config_body.at("prompt").get<std::string>();
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
            if (!config_body.contains("system_prompt") && config_body.contains("prompt")) {
                mode_config_.system_prompt = config_body.at("prompt").get<std::string>();
            }

            CONFIG_AUTO_SET(file_body["mode_param"], template_filename_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], axmodel_num);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_post_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], post_config_path);

            CONFIG_AUTO_SET(file_body["mode_param"], tokenizer_type);
            CONFIG_AUTO_SET(file_body["mode_param"], url_tokenizer_model);
            CONFIG_AUTO_SET(file_body["mode_param"], b_bos);
            CONFIG_AUTO_SET(file_body["mode_param"], b_eos);

            CONFIG_AUTO_SET(file_body["mode_param"], full_attention_interval);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_tokens_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], tokens_embed_num);
            CONFIG_AUTO_SET(file_body["mode_param"], tokens_embed_size);
            CONFIG_AUTO_SET(file_body["mode_param"], b_use_mmap_load_embed);

            CONFIG_AUTO_SET(file_body["mode_param"], filename_image_encoder_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_cache_dir);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_width);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_height);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_patch_size);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_temporal_patch_size);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_spatial_merge_size);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_fps);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_tokens_per_second);

            CONFIG_AUTO_SET(file_body["mode_param"], prefill_token_num);
            CONFIG_AUTO_SET(file_body["mode_param"], prefill_max_token_num);
            CONFIG_AUTO_SET(file_body["mode_param"], prefill_grpid);
            CONFIG_AUTO_SET(file_body["mode_param"], max_token_len);
            CONFIG_AUTO_SET(file_body["mode_param"], kv_cache_num);
            CONFIG_AUTO_SET(file_body["mode_param"], kv_cache_size);
            CONFIG_AUTO_SET(file_body["mode_param"], b_use_mmap_load_layer);

            CONFIG_AUTO_SET(file_body["mode_param"], vision_cache_dir);

            CONFIG_AUTO_SET(file_body["mode_param"], vision_width);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_height);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_temporal_patch_size);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_spatial_merge_size);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_patch_size);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_fps);
            CONFIG_AUTO_SET(file_body["mode_param"], vision_tokens_per_second);

            const auto parse_vlm_type = [&](const nlohmann::json &obj, const char *key) -> bool {
                if (!obj.contains(key)) {
                    return false;
                }

                const auto &value = obj[key];
                std::optional<VLMType> parsed;
                if (value.is_number_integer()) {
                    parsed = VLMTypeFromInt(value.get<int>());
                } else if (value.is_string()) {
                    parsed = VLMTypeFromString(value.get<std::string>());
                } else {
                    SLOGE("%s must be int or string. choices: %s", key, VLMTypeChoices().c_str());
                    throw std::runtime_error("invalid vlm_type");
                }

                if (!parsed.has_value()) {
                    SLOGE("invalid %s value. choices: %s", key, VLMTypeChoices().c_str());
                    throw std::runtime_error("invalid vlm_type");
                }

                mode_config_.vlm_type = *parsed;
                return true;
            };

            if (!parse_vlm_type(config_body, "vlm_type") && !parse_vlm_type(config_body, "VLM_TYPE")) {
                parse_vlm_type(file_body["mode_param"], "vlm_type") ||
                    parse_vlm_type(file_body["mode_param"], "VLM_TYPE");
            }

            mode_config_.template_filename_axmodel      = base_model + mode_config_.template_filename_axmodel;
            mode_config_.filename_post_axmodel          = base_model + mode_config_.filename_post_axmodel;
            mode_config_.filename_tokens_embed          = base_model + mode_config_.filename_tokens_embed;
            mode_config_.url_tokenizer_model            = base_model + mode_config_.url_tokenizer_model;
            mode_config_.post_config_path               = base_model + mode_config_.post_config_path;
            mode_config_.filename_image_encoder_axmodel = base_model + mode_config_.filename_image_encoder_axmodel;
            mode_config_.vision_cache_dir               = base_model + mode_config_.vision_cache_dir;
            mode_config_.runing_callback                = [this](std::string str, float token_per_sec, void *reserve) {
                if (this->out_callback_) {
                    this->out_callback_(str, false);
                }
            };
            lLaMa_ = std::make_unique<LLM>();
            if (!lLaMa_->Init(mode_config_)) {
                lLaMa_->Deinit();
                lLaMa_.reset();
                return -2;
            }

        } catch (...) {
            SLOGE("config false");
            return -3;
        }
        return 0;
    }

    void run()
    {
        for (;;) {
            auto par = async_list_.get();
            if (par.prompt.empty() && par.image_paths.empty()) break;
            inference(par);
        }
    }

    bool stage_image(const std::string &image_bytes)
    {
        const std::string temp_file = save_image_to_tempfile(image_bytes);
        if (temp_file.empty()) {
            return false;
        }

        std::lock_guard<std::mutex> lock(pending_media_mutex_);
        pending_image_paths_.push_back(temp_file);
        pending_temp_files_.push_back(temp_file);
        return true;
    }

    int inference_async(const std::string &msg)
    {
        if (msg.empty()) return -1;
        if (async_list_.size() < 3) {
            inference_async_par par;
            par.prompt = msg;
            {
                std::lock_guard<std::mutex> lock(pending_media_mutex_);
                par.image_paths = std::move(pending_image_paths_);
                par.temp_files  = std::move(pending_temp_files_);
                pending_image_paths_.clear();
                pending_temp_files_.clear();
            }
            async_list_.put(par);
        } else {
            SLOGE("inference list is full\n");
        }
        return async_list_.size();
    }

    void inference(const inference_async_par &request)
    {
        try {
            if (lLaMa_) {
                std::vector<Content> history;
                if (!mode_config_.system_prompt.empty()) {
                    history.push_back({SYSTEM, TEXT, mode_config_.system_prompt});
                }

                Content user{USER, request.image_paths.empty() ? TEXT : IMAGE, request.prompt};
                history.push_back(user);

                if (!request.image_paths.empty()) {
                    std::vector<MediaInputs> media_inputs;
                    media_inputs.push_back({history.size() - 1, request.image_paths});
                    history = lLaMa_->Run(history, media_inputs);
                } else {
                    history = lLaMa_->Run(history);
                }

                std::string out;
                if (!history.empty() && history.back().role == ASSISTANT) {
                    out = history.back().data;
                }

                if (out_callback_) out_callback_(out, true);
            }
        } catch (...) {
            SLOGW("lLaMa_->Run have error!");
        }

        cleanup_temp_files(request.temp_files);
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
        cleanup_pending_media();
        if (lLaMa_) lLaMa_->Deinit();
        if (lLaMa_) lLaMa_.reset();
        return true;
    }

    static unsigned int getNextPort()
    {
        unsigned int port = next_port_++;
        if (port > 8089) {
            next_port_ = 8080;
            port       = 8080;
        }
        return port;
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

    llm_task(const std::string &workid) : tokenizer_server_flage_(false), port_(getNextPort())
    {
        inference_run_ = std::make_unique<std::thread>(std::bind(&llm_task::run, this));
        _ax_init();
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
            inference_async_par par;
            async_list_.put(par);
            if (lLaMa_) lLaMa_->Stop();
            inference_run_->join();
            inference_run_.reset();
        }
        cleanup_pending_media();
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
        cleanup_pending_media();
        _ax_deinit();
    }
};

std::atomic<unsigned int> llm_task::next_port_{8080};
int llm_task::ax_init_flage_ = 0;

#undef CONFIG_AUTO_SET

class llm_llm : public StackFlow {
private:
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_llm() : StackFlow("llm2")
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
    }

    void pause(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_llm2::work:%s", data.c_str());

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
            if (!llm_task_obj->stage_image(*next_data)) {
                error_body["code"]    = -26;
                error_body["message"] = "Image staging failed.";
                send("None", "None", error_body, unit_name_);
            }
            return;
        }
        llm_task_obj->inference_async(sample_unescapeString(*next_data));
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
                llm_task_obj->inference_async(sample_json_str_get(data, "delta"));
            }
        } else {
            llm_task_obj->inference_async(data);
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

            llm_task_obj->set_output(std::bind(&llm_llm::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("llm") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_llm::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if ((input.find("asr") != std::string::npos) || (input.find("whisper") != std::string::npos)) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_llm::task_asr_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                } else if (input.find("kws") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_llm::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
        SLOGI("llm_llm2::link:%s", data.c_str());
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
                std::bind(&llm_llm::task_asr_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("kws") != std::string::npos) {
            ret = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_llm::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
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
        SLOGI("llm_llm2::unlink:%s", data.c_str());
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
        SLOGI("llm_llm2::taskinfo:%s", data.c_str());
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
        SLOGI("llm_llm2::exit:%s", data.c_str());

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

    ~llm_llm()
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
    llm_llm llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}