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
#include <fstream>
#include <stdexcept>
#include <iostream>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "camera.h"
#include "axera_camera.h"
#include <glob.h>
#include <opencv2/opencv.hpp>

using namespace StackFlows;
int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    main_exit_flage = 1;
}

typedef std::function<void(const void *, int)> task_callback_t;

typedef camera_t* (*hal_camera_open_fun)(const char* pdev_name, int width, int height, int fps);
typedef int (*hal_camera_close_fun)(camera_t* camera);


#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

class llm_task {
private:
    camera_t *cam;
    hal_camera_open_fun hal_camera_open;
    hal_camera_close_fun hal_camera_close;
public:
    std::string response_format_;
    task_callback_t out_callback_;
    bool enoutput_;
    bool enstream_;
    std::atomic_int cap_status_;
    std::unique_ptr<std::thread> camera_cap_thread_;
    std::atomic_bool camera_clear_flage_;

    std::string devname_;
    int frame_width_;
    int frame_height_;
    cv::Mat yuv_dist_;

    static void on_cap_fream(void *pData, uint32_t width, uint32_t height, uint32_t Length, void *ctx)
    {
        llm_task *self = static_cast<llm_task *>(ctx);
        int src_offsetX;
        int src_offsetY;
        int src_W;
        int src_H;
        int dst_offsetX;
        int dst_offsetY;
        int dst_W;
        int dst_H;
        if ((self->frame_height_ == height) && (self->frame_width_ == width)) {
            if (self->out_callback_) self->out_callback_(pData, Length);
        } else {
            if ((self->frame_height_ >= height) && (self->frame_width_ >= width)) {
                src_offsetX = 0;
                src_offsetY = 0;
                src_W       = width;
                src_H       = height;
                dst_offsetX = (self->frame_width_ == src_W) ? 0 : (self->frame_width_ - src_W) / 2;
                dst_offsetY = (self->frame_height_ == src_H) ? 0 : (self->frame_height_ - src_H) / 2;
                dst_W       = width;
                dst_H       = height;
            } else if ((self->frame_height_ <= height) && (self->frame_width_ <= width)) {
                src_offsetX = (self->frame_width_ == width) ? 0 : (width - self->frame_width_) / 2;
                src_offsetY = (self->frame_height_ == height) ? 0 : (height - self->frame_height_) / 2;
                src_W       = self->frame_width_;
                src_H       = self->frame_height_;
                dst_offsetX = 0;
                dst_offsetY = 0;
                dst_W       = src_W;
                dst_H       = src_H;
            } else if ((self->frame_height_ >= height) && (self->frame_width_ <= width)) {
                src_offsetX = (self->frame_width_ == width) ? 0 : (width - self->frame_width_) / 2;
                src_offsetY = 0;
                src_W       = self->frame_width_;
                src_H       = height;
                dst_offsetX = 0;
                dst_offsetY = (self->frame_height_ == src_H) ? 0 : (self->frame_height_ - src_H) / 2;
                dst_W       = src_W;
                dst_H       = src_H;
            } else {
                src_offsetX = 0;
                src_offsetY = (self->frame_height_ == height) ? 0 : (height - self->frame_height_) / 2;
                src_W       = width;
                src_H       = self->frame_height_;
                dst_offsetX = (self->frame_width_ == src_W) ? 0 : (self->frame_width_ - src_W) / 2;
                dst_offsetY = 0;
                dst_W       = src_W;
                dst_H       = src_H;
            }
            cv::Mat yuv_src(height, width, CV_8UC2, pData);
            yuv_src(cv::Rect(src_offsetX, src_offsetY, src_W, src_H))
                .copyTo(self->yuv_dist_(cv::Rect(dst_offsetX, dst_offsetY, dst_W, dst_H)));
            if (self->out_callback_)
                self->out_callback_(self->yuv_dist_.data, self->frame_height_ * self->frame_width_ * 2);
        }
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");
            devname_         = config_body.at("input");
            frame_width_     = config_body.at("frame_width");
            frame_height_    = config_body.at("frame_height");

        } catch (...) {
            return true;
        }
        enstream_ = (response_format_.find("stream") != std::string::npos);
        yuv_dist_ = cv::Mat(frame_height_, frame_width_, CV_8UC2, cv::Scalar(0, 128));
        if(devname_.find("/dev/video") != std::string::npos){
            hal_camera_open = camera_open;
            hal_camera_close = camera_close;
        }else if(devname_.find("axera_") != std::string::npos){
            hal_camera_open = axera_camera_open;
            hal_camera_close = axera_camera_close;
        }else {
            return true;
        }
        return false;
    }

    int load_model(const nlohmann::json &config_body)
    {
        if (parse_config(config_body)) {
            return -1;
        }
        try {
            cam = hal_camera_open(devname_.c_str(), frame_width_, frame_height_, 30);
            if (cam == NULL) {
                printf("Camera open failed \n");
                return -1;
            }
            cam->ctx_ = static_cast<void *>(this);
            cam->camera_capture_callback_set(cam, on_cap_fream);
            cam->camera_capture_start(cam);
        } catch (...) {
            SLOGE("config file read false");
            return -3;
        }
        return 0;
    }

    void inference(const std::string &msg)
    {
        // std::cout << msg << std::endl;
        if (out_callback_) out_callback_("None", 4);
    }

    llm_task(const std::string &workid)
    {
        cam = NULL;
    }

    ~llm_task()
    {
        if (cam) {
            cam->camera_capture_stop(cam);
            hal_camera_close(cam);
            cam = NULL;
        }
    }
};

class llm_camera : public StackFlow {
private:
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_camera() : StackFlow("camera")
    {
        rpc_ctx_->register_rpc_action(
            "list_camera", std::bind(&llm_camera::list_camera, this, std::placeholders::_1, std::placeholders::_2));
    }

    std::string list_camera(pzmq *_pzmq, const std::string &rawdata)
    {
        nlohmann::json req_body;
        std::string zmq_url    = RPC_PARSE_TO_FIRST(rawdata);
        std::string param_json = RPC_PARSE_TO_SECOND(rawdata);
        std::vector<std::string> devices;
        glob_t glob_result;
        glob("/dev/video*", GLOB_TILDE, NULL, &glob_result);
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            devices.push_back(std::string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
        send("camera.devices", devices, LLM_NO_ERROR, sample_json_str_get(param_json, "work_id"), zmq_url);
        return LLM_NONE;
    }

    void task_output(const std::weak_ptr<llm_task> llm_task_obj_weak,
                     const std::weak_ptr<llm_channel_obj> llm_channel_weak, const void *data, int size)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::string out_data((char *)data, size);
        llm_channel->send_raw_to_pub(out_data);
        if (llm_task_obj->enoutput_) {
            std::string base64_data;
            int ret = StackFlows::encode_base64(out_data, base64_data);
            std::string out_json_str;
            out_json_str.reserve(llm_channel->request_id_.size() + llm_channel->work_id_.size() + base64_data.size() + 108);
            out_json_str += R"({"request_id":")";
            out_json_str += llm_channel->request_id_;
            out_json_str += R"(","work_id":")";
            out_json_str += llm_channel->work_id_;
            out_json_str += R"(","object":"image.yuvraw.base64","error":{"code":0, "message":""},"data":")";
            out_json_str += base64_data;
            out_json_str += "\"}\n";
            llm_channel->send_raw_to_usr(out_json_str);
        }
    }

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        int work_id_num   = sample_get_work_id_num(work_id);
        auto llm_channel  = get_channel(work_id);
        auto llm_task_obj = std::make_shared<llm_task>(work_id);
        nlohmann::json config_body;
        try {
            config_body = nlohmann::json::parse(data);
        } catch (...) {
            error_body["code"]    = -2;
            error_body["message"] = "json format error.";
            send("None", "None", error_body, unit_name_);
            return -2;
        }
        int ret = llm_task_obj->load_model(config_body);
        if (ret == 0) {
            llm_channel->set_output(llm_task_obj->enoutput_);
            llm_channel->set_stream(llm_task_obj->enstream_);
            llm_task_obj->set_output(std::bind(&llm_camera::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));
            llm_task_[work_id_num] = llm_task_obj;
            send("None", "None", LLM_NO_ERROR, work_id);
            return 0;
        } else {
            error_body["code"]    = -5;
            error_body["message"] = "Model loading failed.";
            send("None", "None", error_body, unit_name_);
            return -1;
        }
    }

    void taskinfo(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("camera.tasklist", req_body, LLM_NO_ERROR, work_id);
        } else {
            if (llm_task_.find(work_id_num) == llm_task_.end()) {
                req_body["code"]    = -6;
                req_body["message"] = "Unit Does Not Exist";
                send("None", "None", req_body, work_id);
                return;
            }
            auto llm_task_obj           = llm_task_[work_id_num];
            req_body["response_format"] = llm_task_obj->response_format_;
            req_body["enoutput"]        = llm_task_obj->enoutput_;
            req_body["input"]           = llm_task_obj->devname_;
            req_body["frame_width"]     = llm_task_obj->frame_width_;
            req_body["frame_height"]    = llm_task_obj->frame_height_;
            send("camera.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return -1;
        }
        auto llm_channel = get_channel(work_id_num);
        llm_channel->stop_subscriber("");
        llm_task_.erase(work_id_num);
        send("None", "None", LLM_NO_ERROR, work_id);
        return 0;
    }

    ~llm_camera()
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
    llm_camera llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}
