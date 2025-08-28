/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "AxclWrapper.h"
#include "base/common.hpp"
#include <ax_sys_api.h>
#include <sys/stat.h>
#include <fstream>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "thread_safe_list.h"

using namespace StackFlows;

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_depth_anything will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;
const char *CONFIG_FILE_DEFAULT = "/usr/local/axcl/axcl.json";

typedef struct {
    std::string depth_anything_model;
    std::string model_type = "segment";
    int img_h              = 640;
    int img_w              = 640;
    uint32_t npu_type      = 0;
} depth_anything_config;

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

class llm_task {
private:
public:
    depth_anything_config mode_config_;
    std::string model_;
    std::unique_ptr<AxclWrapper> depth_anything_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    std::vector<unsigned char> image_data_;
    bool enoutput_;
    bool enstream_;
    static int ax_init_flage_;
    task_callback_t out_callback_;
    std::atomic_bool camera_flage_;
    std::unique_ptr<std::thread> inference_run_;
    thread_safe::list<inference_async_par> async_list_;

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
            } else
                throw std::string("error");
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
            CONFIG_AUTO_SET(file_body["mode_param"], depth_anything_model);
            CONFIG_AUTO_SET(file_body["mode_param"], img_h);
            CONFIG_AUTO_SET(file_body["mode_param"], img_w);
            CONFIG_AUTO_SET(file_body["mode_param"], model_type);
            CONFIG_AUTO_SET(file_body["mode_param"], npu_type);
            mode_config_.depth_anything_model = base_model + mode_config_.depth_anything_model;
            depth_anything_                   = std::make_unique<AxclWrapper>();
            if (!depth_anything_->initialize(CONFIG_FILE_DEFAULT, 0, 0, mode_config_.depth_anything_model, true, true, 0, 0)) {
                SLOGE("Init depth_anything_model model failed!\n");
                return -5;
            }
        } catch (...) {
            SLOGE("config false");
            return -6;
        }
        return 0;
    }

    std::string format_float(double value, int decimal_places)
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(decimal_places) << value;
        return out.str();
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

    bool inference_decode(const std::string &msg)
    {
        cv::Mat src = cv::imdecode(std::vector<uint8_t>(msg.begin(), msg.end()), cv::IMREAD_COLOR);
        if (src.empty()) return true;
        return inference_async(src) ? false : true;
    }

    bool inference_raw_yuv(const std::string &msg)
    {
        if (msg.size() != mode_config_.img_w * mode_config_.img_h * 2) {
            throw std::string("img size error");
        }
        cv::Mat camera_data(mode_config_.img_h, mode_config_.img_w, CV_8UC2, (void *)msg.data());
        cv::Mat rgb;
        cv::cvtColor(camera_data, rgb, cv::COLOR_YUV2RGB_YUYV);
        return inference_async(rgb, true) ? false : true;
    }

    bool inference_raw_rgb(const std::string &msg)
    {
        if (msg.size() != mode_config_.img_w * mode_config_.img_h * 3) {
            throw std::string("img size error");
        }
        cv::Mat camera_data(mode_config_.img_h, mode_config_.img_w, CV_8UC3, (void *)msg.data());
        return inference_async(camera_data, false) ? false : true;
    }

    bool inference_raw_bgr(const std::string &msg)
    {
        if (msg.size() != mode_config_.img_w * mode_config_.img_h * 3) {
            throw std::string("img size error");
        }
        cv::Mat camera_data(mode_config_.img_h, mode_config_.img_w, CV_8UC3, (void *)msg.data());
        return inference_async(camera_data) ? false : true;
    }

    void run()
    {
        inference_async_par par;
        for (;;) {
            {
                par = async_list_.get();
                if (par.inference_src.empty()) break;
                inference(par.inference_src, par.inference_bgr2rgb);
            }
        }
    }

    int inference_async(cv::Mat &src, bool bgr2rgb = true)
    {
        if (async_list_.size() < 3) {
            inference_async_par par;
            par.inference_src     = src.clone();
            par.inference_bgr2rgb = bgr2rgb;
            async_list_.put(par);
        } else {
            SLOGE("inference list is full\n");
        }
        return async_list_.size();
    }

    bool inference(cv::Mat &src, bool bgr2rgb = true)
    {
        try {
            int ret = -1;
            depth_anything_->set();
            std::vector<uint8_t> image(mode_config_.img_w * mode_config_.img_h * 3, 0);
            common::get_input_data_no_letterbox(src, image, mode_config_.img_h, mode_config_.img_w, true);
            axclrtMemcpy(depth_anything_->getInputPointer(0), image.data(), image.size(),
                          AXCL_MEMCPY_HOST_TO_DEVICE);
            if (!depth_anything_->run(false)) {
                SLOGE("Run depth_anything model failed!\n");
                throw std::string("depth_anything_ RunSync error");
            }
            std::string depth_anything_output;
            depth_anything_->post_process(src, mode_config_.img_h, mode_config_.img_w, mode_config_.model_type, depth_anything_output);
            if (out_callback_) out_callback_(depth_anything_output, true);
        } catch (...) {
            SLOGW("depth_anything_->Run have error!");
            return true;
        }
        return false;
    }

    llm_task(const std::string &workid)
    {
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
            inference_async_par par;
            async_list_.put(par);
            inference_run_->join();
            inference_run_.reset();
        }
    }

    ~llm_task()
    {
        stop();
        if (depth_anything_) depth_anything_->finalize();
    }
};
int llm_task::ax_init_flage_ = 0;
#undef CONFIG_AUTO_SET

class llm_depth_anything : public StackFlow {
private:
    int task_count_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_depth_anything() : StackFlow("depth_anything")
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
        int len = encode_base64(data, base64_data);
        if (llm_channel->enstream_) {
            static int count = 0;
            nlohmann::json data_body;
            data_body["index"] = count++;
            // if (!finish)
            data_body["delta"] = base64_data;
            // else
            // data_body["delta"] = std::string("");
            data_body["finish"] = finish;
            if (finish) count = 0;
            llm_channel->send(llm_task_obj->response_format_, data_body, LLM_NO_ERROR);
        } else if (finish) {
            // SLOGI("send utf-8");
            llm_channel->send(llm_task_obj->response_format_, base64_data, LLM_NO_ERROR);
        }
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
        if (data.empty() || (data == "None")) {
            error_body["code"]    = -24;
            error_body["message"] = "The inference data is empty.";
            send("None", "None", error_body, unit_name_);
            return;
        }
        const std::string *next_data = &data;
        bool enstream                = (object.find("stream") == std::string::npos) ? false : true;
        int ret;
        std::string tmp_msg1;
        if (enstream) {
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
        // must encode base64
        std::string tmp_msg2;
        ret = decode_base64((*next_data), tmp_msg2);
        if (ret == -1) {
            error_body["code"]    = -23;
            error_body["message"] = "Base64 decoding error.";
            send("None", "None", error_body, unit_name_);
            return;
        }
        next_data = &tmp_msg2;

        if (llm_task_obj->inference_decode(*next_data)) {
            error_body["code"]    = -11;
            error_body["message"] = "Model run failed.";
            send("None", "None", error_body, unit_name_);
        }
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
            send("None", "None", error_body, unit_name_);
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
            send("None", "None", error_body, unit_name_);
            return -2;
        }
        int ret = llm_task_obj->load_model(config_body);
        if (ret == 0) {
            llm_channel->set_output(llm_task_obj->enoutput_);
            llm_channel->set_stream(llm_task_obj->enstream_);

            llm_task_obj->set_output(std::bind(&llm_depth_anything::task_output, this, llm_task_obj, llm_channel,
                                               std::placeholders::_1, std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("depth_anything") != std::string::npos) {
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_depth_anything::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else {
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
                llm_task_[work_id_num] = llm_task_obj;
                SLOGI("load_mode success");
                send("None", "None", LLM_NO_ERROR, work_id);
                return 0;
            }
            return 0;
        } else {
            SLOGE("load_mode Failed");
            error_body["code"]    = -5;
            error_body["message"] = "Model loading failed.";
            send("None", "None", error_body, unit_name_);
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_depth_anything::link:%s", data.c_str());
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
        if (data.find("depth_anything") != std::string::npos) {
            ret = llm_channel->subscriber_work_id(
                "",
                std::bind(&llm_depth_anything::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                          std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("camera") != std::string::npos) {
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
        SLOGI("llm_depth_anything::unlink:%s", data.c_str());
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
        SLOGI("llm_depth_anything::taskinfo:%s", data.c_str());
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("depth_anything.tasklist", req_body, LLM_NO_ERROR, work_id);
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
            send("depth_anything.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_depth_anything::exit:%s", data.c_str());

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

    ~llm_depth_anything()
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

int main()
{
    signal(SIGTERM, __sigint);
    signal(SIGINT, __sigint);
    mkdir("/tmp/llm", 0777);
    llm_depth_anything llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}