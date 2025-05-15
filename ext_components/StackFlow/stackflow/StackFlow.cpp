/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include "sample_log.h"

using namespace StackFlows;

std::string llm_channel_obj::uart_push_url;

// Deprecated
#define RPC_PARSE_TO_PARAM_OLD(obj) \
    sample_json_str_get(obj, "zmq_com"), sample_unescapeString(sample_json_str_get(obj, "raw_data"))

#define RPC_PARSE_TO_PARAM(obj)     RPC_PARSE_TO_FIRST(obj), RPC_PARSE_TO_SECOND(obj)
#define PTR_RPC_PARSE_TO_PARAM(obj) PTR_RPC_PARSE_TO_FIRST(obj), PTR_RPC_PARSE_TO_SECOND(obj)

llm_channel_obj::llm_channel_obj(const std::string &_publisher_url, const std::string &inference_url,
                                 const std::string &unit_name)
    : unit_name_(unit_name), inference_url_(inference_url)
{
    zmq_url_index_ = -1000;
    zmq_[-1]       = std::make_shared<pzmq>(_publisher_url, ZMQ_PUB);
    zmq_[-2].reset();
}

llm_channel_obj::~llm_channel_obj()
{
}

void llm_channel_obj::subscriber_event_call(const std::function<void(const std::string &, const std::string &)> &call,
                                            pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw)
{
    auto _raw                            = raw->string();
    const char *user_inference_flage_str = "\"action\"";
    std::size_t pos                      = _raw.find(user_inference_flage_str);
    while (true) {
        if (pos == std::string::npos) {
            break;
        } else if ((pos > 0) && (_raw[pos - 1] != '\\')) {
            std::string zmq_com = sample_json_str_get(_raw, "zmq_com");
            if (!zmq_com.empty()) set_push_url(zmq_com);
            request_id_ = sample_json_str_get(_raw, "request_id");
            work_id_    = sample_json_str_get(_raw, "work_id");
            break;
        }
        pos = _raw.find(user_inference_flage_str, pos + sizeof(user_inference_flage_str));
    }
    call(sample_json_str_get(_raw, "object"), sample_json_str_get(_raw, "data"));
}

int llm_channel_obj::subscriber_work_id(const std::string &work_id,
                                        const std::function<void(const std::string &, const std::string &)> &call)
{
    int id_num;
    std::string subscriber_url;
    std::regex pattern(R"((\w+)\.(\d+))");
    std::smatch matches;
    if ((!work_id.empty()) && std::regex_match(work_id, matches, pattern)) {
        if (matches.size() == 3) {
            // std::string part1 = matches[1].str();
            id_num                     = std::stoi(matches[2].str());
            std::string input_url_name = work_id + ".out_port";
            std::string input_url      = unit_call("sys", "sql_select", input_url_name);
            if (input_url.empty()) {
                return -1;
            }
            subscriber_url = input_url;
        }
    } else {
        id_num         = 0;
        subscriber_url = inference_url_;
    }
    zmq_[id_num] = std::make_shared<pzmq>(
        subscriber_url, ZMQ_SUB,
        std::bind(&llm_channel_obj::subscriber_event_call, this, call, std::placeholders::_1, std::placeholders::_2));
    return 0;
}

void llm_channel_obj::stop_subscriber_work_id(const std::string &work_id)
{
    int id_num;
    std::regex pattern(R"((\w+)\.(\d+))");
    std::smatch matches;
    if (std::regex_match(work_id, matches, pattern)) {
        if (matches.size() == 3) {
            // std::string part1 = matches[1].str();
            id_num = std::stoi(matches[2].str());
        }
    } else {
        id_num = 0;
    }
    if (zmq_.find(id_num) != zmq_.end()) zmq_.erase(id_num);
}

void llm_channel_obj::subscriber(const std::string &zmq_url, const pzmq::msg_callback_fun &call)
{
    zmq_url_map_[zmq_url]       = zmq_url_index_--;
    zmq_[zmq_url_map_[zmq_url]] = std::make_shared<pzmq>(zmq_url, ZMQ_SUB, call);
}

void llm_channel_obj::stop_subscriber(const std::string &zmq_url)
{
    if (zmq_url.empty()) {
        zmq_.clear();
        zmq_url_map_.clear();
    } else if (zmq_url_map_.find(zmq_url) != zmq_url_map_.end()) {
        zmq_.erase(zmq_url_map_[zmq_url]);
        zmq_url_map_.erase(zmq_url);
    }
}

int llm_channel_obj::send_raw_to_pub(const std::string &raw)
{
    return zmq_[-1]->send_data(raw);
}

int llm_channel_obj::send_raw_to_usr(const std::string &raw)
{
    if (zmq_[-2]) {
        return zmq_[-2]->send_data(raw);
    } else {
        return -1;
    }
}

void llm_channel_obj::set_push_url(const std::string &url)
{
    if (output_url_ != url) {
        output_url_ = url;
        zmq_[-2].reset(new pzmq(output_url_, ZMQ_PUSH));
    }
}

void llm_channel_obj::cear_push_url()
{
    zmq_[-2].reset();
}

int llm_channel_obj::send_raw_for_url(const std::string &zmq_url, const std::string &raw)
{
    pzmq _zmq(zmq_url, ZMQ_PUSH);
    return _zmq.send_data(raw);
}

int llm_channel_obj::output_to_uart(const std::string &data)
{
    return send_raw_for_url(uart_push_url, data);
}

StackFlow::StackFlow::StackFlow(const std::string &unit_name)
    : work_id_num_cout_(1000), unit_name_(unit_name), rpc_ctx_(std::make_unique<pzmq>(unit_name))
{
    event_queue_.appendListener(EVENT_NONE, std::bind(&StackFlow::_none_event, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_PAUSE, std::bind(&StackFlow::_pause, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_WORK, std::bind(&StackFlow::_work, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_EXIT, std::bind(&StackFlow::_exit, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_SETUP, std::bind(&StackFlow::_setup, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_LINK, std::bind(&StackFlow::_link, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_UNLINK, std::bind(&StackFlow::_unlink, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_TASKINFO, std::bind(&StackFlow::_taskinfo, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_SYS_INIT, std::bind(&StackFlow::_sys_init, this, std::placeholders::_1));
    event_queue_.appendListener(EVENT_REPEAT_EVENT, std::bind(&StackFlow::_repeat_loop, this, std::placeholders::_1));
    rpc_ctx_->register_rpc_action(
        "setup", std::bind(&StackFlow::_rpc_setup, this, std::placeholders::_1, std::placeholders::_2));
    rpc_ctx_->register_rpc_action(
        "pause", std::bind(&StackFlow::_rpc_pause, this, std::placeholders::_1, std::placeholders::_2));
    rpc_ctx_->register_rpc_action("work",
                                  std::bind(&StackFlow::_rpc_work, this, std::placeholders::_1, std::placeholders::_2));
    rpc_ctx_->register_rpc_action("exit",
                                  std::bind(&StackFlow::_rpc_exit, this, std::placeholders::_1, std::placeholders::_2));
    rpc_ctx_->register_rpc_action("link",
                                  std::bind(&StackFlow::_rpc_link, this, std::placeholders::_1, std::placeholders::_2));
    rpc_ctx_->register_rpc_action(
        "unlink", std::bind(&StackFlow::_rpc_unlink, this, std::placeholders::_1, std::placeholders::_2));
    rpc_ctx_->register_rpc_action(
        "taskinfo", std::bind(&StackFlow::_rpc_taskinfo, this, std::placeholders::_1, std::placeholders::_2));

    status_.store(0);
    exit_flage_.store(false);
    even_loop_thread_              = std::make_unique<std::thread>(std::bind(&StackFlow::even_loop, this));
    llm_channel_obj::uart_push_url = std::string("ipc:///tmp/llm/5556.sock");
    status_.store(1);
    repeat_event(1000, [this]() {
        std::string serial_zmq_url = this->sys_sql_select("serial_zmq_url");
        if (!serial_zmq_url.empty()) {
            SLOGI("serial_zmq_url:%s", serial_zmq_url.c_str());
            llm_channel_obj::uart_push_url = serial_zmq_url;
            return 0;
        } else {
            return 1;
        }
    });
}

StackFlow::~StackFlow()
{
    while (1) {
        auto iteam = llm_task_channel_.begin();
        if (iteam == llm_task_channel_.end()) {
            break;
        }
        sys_release_unit(iteam->first, "");
        iteam->second.reset();
        llm_task_channel_.erase(iteam->first);
    }
    exit_flage_.store(true);
    event_queue_.enqueue(EVENT_NONE, nullptr);
    even_loop_thread_->join();
}

void StackFlow::even_loop()
{
    while (!exit_flage_.load()) {
        event_queue_.wait();
        event_queue_.process();
    }
}

void StackFlow::_none_event(const std::shared_ptr<void> &arg)
{
    // std::shared_ptr<stackflow_data> originalPtr = std::static_pointer_cast<stackflow_data>(arg);
}

void StackFlow::_sys_init(const std::shared_ptr<void> &arg)
{
    // todo:...
}

std::string StackFlow::_rpc_setup(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data)
{
    event_queue_.enqueue(EVENT_SETUP, std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
    return std::string("None");
}

int StackFlow::setup(const std::string &zmq_url, const std::string &raw)
{
    SLOGI("StackFlow::setup raw zmq_url:%s raw:%s", zmq_url.c_str(), raw.c_str());
    int workid_num      = sys_register_unit(unit_name_);
    std::string work_id = unit_name_ + "." + std::to_string(workid_num);
    auto task_channel   = get_channel(workid_num);
    task_channel->set_push_url(zmq_url);
    task_channel->request_id_ = sample_json_str_get(raw, "request_id");
    task_channel->work_id_    = work_id;
    if (setup(work_id, sample_json_str_get(raw, "object"), sample_json_str_get(raw, "data"))) {
        sys_release_unit(workid_num, work_id);
    }
    return 0;
}

int StackFlow::setup(const std::string &work_id, const std::string &object, const std::string &data)
{
    SLOGI("StackFlow::setup");
    if (_setup_) {
        return _setup_(work_id, object, data);
    }
    nlohmann::json error_body;
    error_body["code"]    = -18;
    error_body["message"] = "not have unit action!";
    send("None", "None", error_body, work_id);
    return -1;
}

std::string StackFlow::_rpc_link(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data)
{
    event_queue_.enqueue(EVENT_LINK, std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
    return std::string("None");
}

void StackFlow::link(const std::string &zmq_url, const std::string &raw)
{
    SLOGI("StackFlow::link raw");
    std::string work_id = sample_json_str_get(raw, "work_id");
    try {
        auto task_channel = get_channel(sample_get_work_id_num(work_id));
        task_channel->set_push_url(zmq_url);
    } catch (...) {
    }
    link(work_id, sample_json_str_get(raw, "object"), sample_json_str_get(raw, "data"));
}

void StackFlow::link(const std::string &work_id, const std::string &object, const std::string &data)
{
    SLOGI("StackFlow::link");
    if (_link_) {
        _link_(work_id, object, data);
        return;
    }
    nlohmann::json error_body;
    error_body["code"]    = -18;
    error_body["message"] = "not have unit action!";
    send("None", "None", error_body, work_id);
}

std::string StackFlow::_rpc_unlink(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data)
{
    event_queue_.enqueue(EVENT_UNLINK, std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
    return std::string("None");
}

void StackFlow::unlink(const std::string &zmq_url, const std::string &raw)
{
    SLOGI("StackFlow::unlink raw");
    std::string work_id = sample_json_str_get(raw, "work_id");
    try {
        auto task_channel = get_channel(sample_get_work_id_num(work_id));
        task_channel->set_push_url(zmq_url);
    } catch (...) {
    }
    unlink(work_id, sample_json_str_get(raw, "object"), sample_json_str_get(raw, "data"));
}

void StackFlow::unlink(const std::string &work_id, const std::string &object, const std::string &data)
{
    SLOGI("StackFlow::unlink");
    if (_unlink_) {
        _unlink_(work_id, object, data);
        return;
    }
    nlohmann::json error_body;
    error_body["code"]    = -18;
    error_body["message"] = "not have unit action!";
    send("None", "None", error_body, work_id);
}

std::string StackFlow::_rpc_work(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data)
{
    event_queue_.enqueue(EVENT_WORK, std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
    return std::string("None");
}

void StackFlow::work(const std::string &zmq_url, const std::string &raw)
{
    SLOGI("StackFlow::work raw");
    std::string work_id = sample_json_str_get(raw, "work_id");
    try {
        auto task_channel = get_channel(sample_get_work_id_num(work_id));
        task_channel->set_push_url(zmq_url);
    } catch (...) {
    }
    work(work_id, sample_json_str_get(raw, "object"), sample_json_str_get(raw, "data"));
}

void StackFlow::work(const std::string &work_id, const std::string &object, const std::string &data)
{
    SLOGI("StackFlow::work");
    if (_work_) {
        _work_(work_id, object, data);
        return;
    }
    nlohmann::json error_body;
    error_body["code"]    = -18;
    error_body["message"] = "not have unit action!";
    send("None", "None", error_body, work_id);
}

std::string StackFlow::_rpc_exit(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data)
{
    event_queue_.enqueue(EVENT_EXIT, std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
    return std::string("None");
}

int StackFlow::exit(const std::string &zmq_url, const std::string &raw)
{
    SLOGI("StackFlow::exit raw");
    std::string work_id = sample_json_str_get(raw, "work_id");
    try {
        auto task_channel = get_channel(sample_get_work_id_num(work_id));
        task_channel->set_push_url(zmq_url);
    } catch (...) {
    }
    if (exit(work_id, sample_json_str_get(raw, "object"), sample_json_str_get(raw, "data")) == 0) {
        return (int)sys_release_unit(-1, work_id);
    }
    return 0;
}

int StackFlow::exit(const std::string &work_id, const std::string &object, const std::string &data)
{
    SLOGI("StackFlow::exit");
    if (_exit_) {
        return _exit_(work_id, object, data);
    }
    nlohmann::json error_body;
    error_body["code"]    = -18;
    error_body["message"] = "not have unit action!";
    send("None", "None", error_body, work_id);
    return 0;
}

std::string StackFlow::_rpc_pause(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data)
{
    event_queue_.enqueue(EVENT_PAUSE, std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
    return std::string("None");
}

void StackFlow::pause(const std::string &zmq_url, const std::string &raw)
{
    SLOGI("StackFlow::pause raw");
    std::string work_id = sample_json_str_get(raw, "work_id");
    try {
        auto task_channel = get_channel(sample_get_work_id_num(work_id));
        task_channel->set_push_url(zmq_url);
    } catch (...) {
    }
    pause(work_id, sample_json_str_get(raw, "object"), sample_json_str_get(raw, "data"));
}

void StackFlow::pause(const std::string &work_id, const std::string &object, const std::string &data)
{
    SLOGI("StackFlow::pause");
    if (_pause_) {
        _pause_(work_id, object, data);
        return;
    }
    nlohmann::json error_body;
    error_body["code"]    = -18;
    error_body["message"] = "not have unit action!";
    send("None", "None", error_body, work_id);
}

std::string StackFlow::_rpc_taskinfo(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data)
{
    event_queue_.enqueue(EVENT_TASKINFO, std::make_shared<stackflow_data>(data->get_param(0), data->get_param(1)));
    return std::string("None");
}

void StackFlow::taskinfo(const std::string &zmq_url, const std::string &raw)
{
    // SLOGI("StackFlow::taskinfo raw");
    std::string work_id = sample_json_str_get(raw, "work_id");
    try {
        auto task_channel = get_channel(sample_get_work_id_num(work_id));
        task_channel->set_push_url(zmq_url);
    } catch (...) {
    }
    taskinfo(work_id, sample_json_str_get(raw, "object"), sample_json_str_get(raw, "data"));
}

void StackFlow::taskinfo(const std::string &work_id, const std::string &object, const std::string &data)
{
    // SLOGI("StackFlow::taskinfo");
    if (_taskinfo_) {
        _taskinfo_(work_id, object, data);
        return;
    }
    nlohmann::json error_body;
    error_body["code"]    = -18;
    error_body["message"] = "not have unit action!";
    send("None", "None", error_body, work_id);
}

int StackFlow::sys_register_unit(const std::string &unit_name)
{
    int work_id_number;
    std::string str_port;
    std::string out_port;
    std::string inference_port;

    unit_call("sys", "register_unit", unit_name, [&](const std::shared_ptr<StackFlows::pzmq_data> &pzmg_msg) {
        str_port       = pzmg_msg->get_param(1);
        out_port       = pzmg_msg->get_param(0, str_port);
        inference_port = pzmg_msg->get_param(1, str_port);
        str_port       = pzmg_msg->get_param(0);
    });
    work_id_number = std::stoi(str_port);
    SLOGI("work_id_number:%d, out_port:%s, inference_port:%s ", work_id_number, out_port.c_str(),
          inference_port.c_str());
    llm_task_channel_[work_id_number] = std::make_shared<llm_channel_obj>(out_port, inference_port, unit_name_);
    return work_id_number;
}

bool StackFlow::sys_release_unit(int work_id_num, const std::string &work_id)
{
    std::string _work_id;
    int _work_id_num;
    if (work_id.empty()) {
        _work_id     = sample_get_work_id(work_id_num, unit_name_);
        _work_id_num = work_id_num;
    } else {
        _work_id     = work_id;
        _work_id_num = sample_get_work_id_num(work_id);
    }
    unit_call("sys", "release_unit", _work_id);
    llm_task_channel_[_work_id_num].reset();
    llm_task_channel_.erase(_work_id_num);
    SLOGI("release work_id %s success", _work_id.c_str());
    return false;
}

std::string StackFlow::sys_sql_select(const std::string &key)
{
    return sample_unescapeString(unit_call("sys", "sql_select", key));
}

void StackFlow::sys_sql_set(const std::string &key, const std::string &val)
{
    nlohmann::json out_body;
    out_body["key"] = key;
    out_body["val"] = val;
    unit_call("sys", "sql_set", out_body.dump());
}

void StackFlow::sys_sql_unset(const std::string &key)
{
    unit_call("sys", "sql_unset", key);
}

void StackFlow::_repeat_loop(const std::shared_ptr<void> &arg)
{
    std::shared_ptr<stackflow_data> originalPtr = std::static_pointer_cast<stackflow_data>(arg);
    std::string action                          = originalPtr->string(0);
    int ms                                      = originalPtr->integer(0);
    repeat_callback_fun_mutex_.lock();
    const auto call_fun = repeat_callback_fun_[action];
    repeat_callback_fun_mutex_.unlock();
    if (call_fun()) {
        int delayms = ms;
        if (delayms)
            std::thread([this, action, delayms, ms]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(delayms));
                this->event_queue_.enqueue(EVENT_REPEAT_EVENT, std::make_shared<stackflow_data>(action, ms));
            }).detach();
        else {
            event_queue_.enqueue(EVENT_REPEAT_EVENT, std::make_shared<stackflow_data>(action, ms));
        }
    } else {
        repeat_callback_fun_mutex_.lock();
        repeat_callback_fun_.erase(action);
        repeat_callback_fun_mutex_.unlock();
    }
}

void StackFlow::repeat_event(int ms, std::function<int(void)> repeat_fun, bool now)
{
    repeat_callback_fun_mutex_.lock();
    std::string action           = std::to_string(repeat_callback_fun_.size() + 1);
    repeat_callback_fun_[action] = repeat_fun;
    repeat_callback_fun_mutex_.unlock();
    if (!now)
        std::thread([this, action, ms]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            this->event_queue_.enqueue(EVENT_REPEAT_EVENT, std::make_shared<stackflow_data>(action, ms));
        }).detach();
    else {
        event_queue_.enqueue(EVENT_REPEAT_EVENT, std::make_shared<stackflow_data>(action, ms));
    }
}
