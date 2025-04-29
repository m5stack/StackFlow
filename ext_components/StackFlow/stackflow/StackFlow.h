/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// #define __cplusplus 1

#include <semaphore.h>
#include <unistd.h>

// #define CONFIG_SUPPORTTHREADSAFE 0

#include <string>
#include <list>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <eventpp/eventqueue.h>
#include <thread>
#include <memory>
#include "json.hpp"
#include <regex>
#include "pzmq.hpp"
#include "StackFlowUtil.h"
namespace StackFlows {
template <typename T>
class ThreadSafeWrapper {
public:
    template <typename... Args>
    ThreadSafeWrapper(Args &&...args) : object(std::make_unique<T>(std::forward<Args>(args)...))
    {
    }

    template <typename Func>
    auto access(Func func) -> decltype(func(std::declval<T &>()))
    {
        std::lock_guard<std::mutex> lock(mutex);
        return func(*object);
    }

    ThreadSafeWrapper &operator=(const T &newValue)
    {
        std::lock_guard<std::mutex> lock(mutex);
        *object = newValue;
        return *this;
    }

    ThreadSafeWrapper &operator=(T &&newValue)
    {
        std::lock_guard<std::mutex> lock(mutex);
        *object = std::move(newValue);
        return *this;
    }

private:
    std::unique_ptr<T> object;
    mutable std::mutex mutex;
};

#define LLM_NO_ERROR std::string("")
#define LLM_NONE     std::string("None")

class llm_channel_obj {
private:
    std::unordered_map<int, std::shared_ptr<pzmq>> zmq_;
    std::atomic<int> zmq_url_index_;
    std::unordered_map<std::string, int> zmq_url_map_;

public:
    std::string unit_name_;
    bool enoutput_;
    bool enstream_;
    std::string request_id_;
    std::string work_id_;
    std::string inference_url_;
    std::string output_url_;
    static std::string uart_push_url;
    std::string publisher_url;

    llm_channel_obj(const std::string &_publisher_url, const std::string &inference_url, const std::string &unit_name);
    ~llm_channel_obj();
    inline void set_output(bool flage)
    {
        enoutput_ = flage;
    }
    inline bool get_output()
    {
        return enoutput_;
    }
    inline void set_stream(bool flage)
    {
        enstream_ = flage;
    }
    inline bool get_stream()
    {
        return enstream_;
    }
    void subscriber_event_call(const std::function<void(const std::string &, const std::string &)> &call, pzmq *_pzmq,
                               const std::shared_ptr<pzmq_data> &raw);
    int subscriber_work_id(const std::string &work_id,
                           const std::function<void(const std::string &, const std::string &)> &call);
    void stop_subscriber_work_id(const std::string &work_id);
    void subscriber(const std::string &zmq_url, const pzmq::msg_callback_fun &call);
    void stop_subscriber(const std::string &zmq_url);
    int check_zmq_errno(void *ctx, void *com, int code);
    int send_raw_to_pub(const std::string &raw);
    int send_raw_to_usr(const std::string &raw);
    template <typename T, typename U>
    int output_data(const std::string &object, const T &data, const U &error_msg)
    {
        return output_data(request_id_, work_id_, object, data, error_msg);
    }
    void set_push_url(const std::string &url);
    void cear_push_url();
    static int output_to_uart(const std::string &data);
    static int send_raw_for_url(const std::string &zmq_url, const std::string &raw);
    template <typename T, typename U>
    static int output_data_for_url(const std::string &zmq_url, const std::string &request_id,
                                   const std::string &work_id, const std::string &object, const U &data,
                                   const T &error_msg, bool outuart = false)
    {
        nlohmann::json out_body;
        out_body["request_id"] = request_id;
        out_body["work_id"]    = work_id;
        out_body["created"]    = time(NULL);
        out_body["object"]     = object;
        out_body["data"]       = data;
        if (error_msg.empty()) {
            out_body["error"]["code"]    = 0;
            out_body["error"]["message"] = "";
        } else
            out_body["error"] = error_msg;
        std::string out = out_body.dump();
        out += "\n";
        if (outuart)
            return output_to_uart(out);
        else
            return send_raw_for_url(zmq_url, out);
    }
    template <typename T, typename U>
    int send(const std::string &object, const U &data, const T &error_msg, const std::string &work_id = "",
             bool outuart = false)
    {
        nlohmann::json out_body;
        out_body["request_id"] = request_id_;
        out_body["work_id"]    = work_id.empty() ? work_id_ : work_id;
        out_body["created"]    = time(NULL);
        out_body["object"]     = object;
        out_body["data"]       = data;
        if (error_msg.empty()) {
            out_body["error"]["code"]    = 0;
            out_body["error"]["message"] = "";
        } else
            out_body["error"] = error_msg;

        std::string out = out_body.dump();
        out += "\n";
        send_raw_to_pub(out);
        if (enoutput_) return send_raw_to_usr(out);
        return 0;
    }
    template <typename T, typename U>
    int output_data(const std::string &request_id, const std::string &work_id, const std::string &object, const T &data,
                    const U &error_msg)
    {
        nlohmann::json out_body;
        out_body["request_id"] = request_id;
        out_body["work_id"]    = work_id;
        out_body["created"]    = time(NULL);
        out_body["object"]     = object;
        out_body["data"]       = data;
        if (error_msg.empty()) {
            out_body["error"]["code"]    = 0;
            out_body["error"]["message"] = "";
        } else
            out_body["error"] = error_msg;

        std::string out = out_body.dump();
        out += "\n";
        send_raw_to_pub(out);
        if (enoutput_) return send_raw_to_usr(out);
        return 0;
    }
};

class stackflow_data {
    union {
        std::string *rawobj;
        std::string *object;
    };
    std::string *data;
};

class StackFlow {
private:
    std::atomic_int work_id_num_cout_;

protected:
    std::string unit_name_;
    typedef enum {
        EVENT_NONE = 0,
        EVENT_SETUP,
        EVENT_EXIT,
        EVENT_PAUSE,
        EVENT_WORK,
        EVENT_STOP,
        EVENT_LINK,
        EVENT_UNLINK,
        EVENT_TASKINFO,
        EVENT_SWITCH_INPUT,
        EVENT_SYS_INIT,
        EVENT_CREAT_CHANNEL,
        EVENT_GET_CHANNL,
        EVENT_REPEAT_EVENT,
        EVENT_EXPORT,
    } local_event_t;

    eventpp::EventQueue<int, void(const std::string &, const std::string &)> event_queue_;
    std::unique_ptr<std::thread> even_loop_thread_;
    std::unique_ptr<pzmq> rpc_ctx_;
    std::atomic<int> status_;
    std::unordered_map<int, std::shared_ptr<llm_channel_obj>> llm_task_channel_;
    std::unordered_map<std::string, std::function<int(void)>> repeat_callback_fun_;
    std::mutex repeat_callback_fun_mutex_;

    void _repeat_loop(const std::string &zmq_url, const std::string &raw);

public:
    std::string request_id_;
    std::string out_zmq_url_;

    std::function<void(const std::string &, const std::string &, const std::string &)> _link_;
    std::function<void(const std::string &, const std::string &, const std::string &)> _unlink_;
    std::function<int(const std::string &, const std::string &, const std::string &)> _setup_;
    std::function<int(const std::string &, const std::string &, const std::string &)> _exit_;
    std::function<void(const std::string &, const std::string &, const std::string &)> _pause_;
    std::function<void(const std::string &, const std::string &, const std::string &)> _work_;
    std::function<void(const std::string &, const std::string &, const std::string &)> _taskinfo_;
    std::function<void(const std::string &, const std::string &, const std::string &)> _get_channl_;

    std::atomic<bool> exit_flage_;

    StackFlow(const std::string &unit_name);
    void even_loop();
    void _none_event(const std::string &data1, const std::string &data2);

    template <typename T>
    std::shared_ptr<llm_channel_obj> get_channel(T workid)
    {
        int _work_id_num;
        if constexpr (std::is_same<T, int>::value) {
            _work_id_num = workid;
        } else if constexpr (std::is_same<T, std::string>::value) {
            _work_id_num = sample_get_work_id_num(workid);
        } else {
            return nullptr;
        }
        return llm_task_channel_.at(_work_id_num);
    }

    std::string _rpc_setup(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _setup(const std::string &zmq_url, const std::string &data)
    {
        // printf("void _setup run \n");
        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) setup(zmq_url, data);
    };
    virtual int setup(const std::string &zmq_url, const std::string &raw);
    virtual int setup(const std::string &work_id, const std::string &object, const std::string &data);

    std::string _rpc_link(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _link(const std::string &zmq_url, const std::string &data)
    {
        // printf("void _link run \n");
        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) link(zmq_url, data);
    };
    virtual void link(const std::string &zmq_url, const std::string &raw);
    virtual void link(const std::string &work_id, const std::string &object, const std::string &data);

    std::string _rpc_unlink(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _unlink(const std::string &zmq_url, const std::string &data)
    {
        // printf("void _unlink run \n");
        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) unlink(zmq_url, data);
    };
    virtual void unlink(const std::string &zmq_url, const std::string &raw);
    virtual void unlink(const std::string &work_id, const std::string &object, const std::string &data);

    std::string _rpc_exit(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _exit(const std::string &zmq_url, const std::string &data)
    {
        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) exit(zmq_url, data);
    }
    virtual int exit(const std::string &zmq_url, const std::string &raw);
    virtual int exit(const std::string &work_id, const std::string &object, const std::string &data);

    std::string _rpc_work(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _work(const std::string &zmq_url, const std::string &data)
    {
        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) work(zmq_url, data);
    }
    virtual void work(const std::string &zmq_url, const std::string &raw);
    virtual void work(const std::string &work_id, const std::string &object, const std::string &data);

    std::string _rpc_pause(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _pause(const std::string &zmq_url, const std::string &data)
    {
        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) pause(zmq_url, data);
    }
    virtual void pause(const std::string &zmq_url, const std::string &raw);
    virtual void pause(const std::string &work_id, const std::string &object, const std::string &data);

    std::string _rpc_taskinfo(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _taskinfo(const std::string &zmq_url, const std::string &data)
    {
        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) taskinfo(zmq_url, data);
    }
    virtual void taskinfo(const std::string &zmq_url, const std::string &raw);
    virtual void taskinfo(const std::string &work_id, const std::string &object, const std::string &data);

    void _sys_init(const std::string &zmq_url, const std::string &data);

    void user_output(const std::string &zmq_url, const std::string &request_id, const std::string &data);
    template <typename T, typename U>
    int send(const std::string &object, const U &data, const T &error_msg, const std::string &work_id,
             const std::string &zmq_url = "")
    {
        nlohmann::json out_body;
        out_body["request_id"] = request_id_;
        out_body["work_id"]    = work_id;
        out_body["created"]    = time(NULL);
        out_body["object"]     = object;
        out_body["data"]       = data;
        if (error_msg.empty()) {
            out_body["error"]["code"]    = 0;
            out_body["error"]["message"] = "";
        } else
            out_body["error"] = error_msg;
        if (zmq_url.empty()) {
            pzmq _zmq(out_zmq_url_, ZMQ_PUSH);
            std::string out = out_body.dump();
            out += "\n";
            return _zmq.send_data(out);
        } else {
            pzmq _zmq(zmq_url, ZMQ_PUSH);
            std::string out = out_body.dump();
            out += "\n";
            return _zmq.send_data(out);
        }
    }

    void llm_firework_exit()
    {
    }

    std::string sys_sql_select(const std::string &key);
    void sys_sql_set(const std::string &key, const std::string &val);
    void sys_sql_unset(const std::string &key);
    int sys_register_unit(const std::string &unit_name);
    template <typename T>
    bool sys_release_unit(T workid)
    {
        std::string _work_id;
        int _work_id_num;
        if constexpr (std::is_same<T, int>::value) {
            _work_id     = sample_get_work_id(workid, unit_name_);
            _work_id_num = workid;
        } else if constexpr (std::is_same<T, std::string>::value) {
            _work_id     = workid;
            _work_id_num = sample_get_work_id_num(workid);
        } else {
            return false;
        }
        pzmq _call("sys");
        _call.call_rpc_action("release_unit", _work_id, [](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data) {});
        llm_task_channel_[_work_id_num].reset();
        llm_task_channel_.erase(_work_id_num);
        // SLOGI("release work_id %s success", _work_id.c_str());
        return false;
    }
    bool sys_release_unit(int work_id_num, const std::string &work_id);
    void repeat_event(int ms, std::function<int(void)> repeat_fun, bool now = true);
    ~StackFlow();
};
};  // namespace StackFlows