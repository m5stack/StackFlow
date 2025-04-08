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

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_sys will be exit!");
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

public:
    LLMAttrType mode_config_;
    std::unique_ptr<LLM> lLaMa_;
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    std::vector<unsigned short> prompt_data_;
    std::vector<unsigned char> image_data_;
    std::vector<unsigned short> img_embed;
    std::string prompt_;
    task_callback_t out_callback_;
    bool enoutput_;
    bool enstream_;

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

            CONFIG_AUTO_SET(file_body["mode_param"], tokenizer_type);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_tokenizer_model);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_tokens_embed);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_post_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], filename_vpm_resampler_axmodedl);
            CONFIG_AUTO_SET(file_body["mode_param"], template_filename_axmodel);
            CONFIG_AUTO_SET(file_body["mode_param"], b_use_topk);
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
            CONFIG_AUTO_SET(file_body["mode_param"], temperature);
            CONFIG_AUTO_SET(file_body["mode_param"], top_p);
            CONFIG_AUTO_SET(file_body["mode_param"], vpm_width);
            CONFIG_AUTO_SET(file_body["mode_param"], vpm_height);

            if (mode_config_.filename_tokenizer_model.find("http:") != std::string::npos) {
                mode_config_.filename_tokenizer_model = "http://localhost:" + std::to_string(port_);
                std::string tokenizer_file;
                if (file_exists(std::string("/opt/m5stack/scripts/") + model_ + std::string("_tokenizer.py"))) {
                    tokenizer_file = std::string("/opt/m5stack/scripts/") + model_ + std::string("_tokenizer.py");
                } else if (file_exists(std::string("/opt/m5stack/scripts/") + std::string("tokenizer_") + model_ +
                                       std::string(".py"))) {
                    tokenizer_file =
                        std::string("/opt/m5stack/scripts/") + std::string("tokenizer_") + model_ + std::string(".py");
                } else {
                    std::string __log = model_ + std::string("_tokenizer.py");
                    __log += " or ";
                    __log += std::string("tokenizer_") + model_ + std::string(".py");
                    __log += " not found!";
                    SLOGE("%s", __log.c_str());
                }
                if (!tokenizer_server_flage_.load()) {
                    tokenizer_pid_ = fork();
                    if (tokenizer_pid_ == 0) {
                        setenv("PYTHONPATH", "/opt/m5stack/lib/vlm/site-packages", 1);
                        execl("/usr/bin/python3", "python3", tokenizer_file.c_str(), "--host", "localhost", "--port",
                              std::to_string(port_).c_str(), "--model_id", (base_model + "tokenizer").c_str(),
                              "--content", ("'" + prompt_ + "'").c_str(), nullptr);
                        perror("execl failed");
                        exit(1);
                    }
                    tokenizer_server_flage_.store(true);
                    SLOGI("port_=%s model_id=%s content=%s", std::to_string(port_).c_str(),
                          (base_model + "tokenizer").c_str(), ("'" + prompt_ + "'").c_str());
                    std::this_thread::sleep_for(std::chrono::seconds(15));
                }
            } else {
                mode_config_.filename_tokenizer_model = base_model + mode_config_.filename_tokenizer_model;
            }
            SLOGI("filename_tokenizer_model: %s", mode_config_.filename_tokenizer_model.c_str());
            mode_config_.filename_tokens_embed           = base_model + mode_config_.filename_tokens_embed;
            mode_config_.filename_post_axmodel           = base_model + mode_config_.filename_post_axmodel;
            mode_config_.filename_vpm_resampler_axmodedl = base_model + mode_config_.filename_vpm_resampler_axmodedl;
            mode_config_.template_filename_axmodel       = base_model + mode_config_.template_filename_axmodel;

            mode_config_.runing_callback = [this](int *p_token, int n_token, const char *p_str, float token_per_sec,
                                                  void *reserve) {
                if (this->out_callback_) {
                    this->out_callback_(std::string(p_str), false);
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

    std::string prompt_complete(const std::string &input)
    {
        std::ostringstream oss_prompt;
        switch (mode_config_.tokenizer_type) {
            case TKT_LLaMa:
                oss_prompt << "<|user|>\n" << input << "</s><|assistant|>\n";
                break;
            case TKT_MINICPM:
                oss_prompt << "<用户>" << input << "<AI>";
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
        // SLOGI("prompt_complete:%s", oss_prompt.str().c_str());
        return oss_prompt.str();
    }

    void inference(const std::string &msg)
    {
        try {
            if (image_data_.empty()) {
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
        } catch (...) {
            SLOGW("lLaMa_->Run have error!");
        }
    }

    bool pause()
    {
        lLaMa_->Stop();
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
        if (port > 8099) {
            next_port_ = 8090;
            port       = 8090;
        }
        return port;
    }

    llm_task(const std::string &workid) : tokenizer_server_flage_(false), port_(getNextPort())
    {
    }

    ~llm_task()
    {
        if (tokenizer_pid_ != -1) {
            kill(tokenizer_pid_, SIGTERM);
            waitpid(tokenizer_pid_, nullptr, WNOHANG);
        }
        if (lLaMa_) {
            lLaMa_->Deinit();
        }
    }
};

std::atomic<unsigned int> llm_task::next_port_{8090};

#undef CONFIG_AUTO_SET

class llm_llm : public StackFlow {
private:
    int task_count_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_llm() : StackFlow("vlm")
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
        if (object.find("jpeg") != std::string::npos) {
            llm_task_obj->image_data_.assign(next_data->begin(), next_data->end());
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
        llm_task_obj->lLaMa_->Stop();
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

            llm_task_obj->set_output(std::bind(&llm_llm::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("vlm") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_llm::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if (input.find("asr") != std::string::npos) {
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
            send("None", "None", error_body, "vlm");
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
        SLOGI("llm_llm::exit:%s", data.c_str());

        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return -1;
        }
        task_pause(llm_task_[work_id_num], get_channel(work_id_num));
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