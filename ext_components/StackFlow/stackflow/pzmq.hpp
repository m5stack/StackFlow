/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <libzmq/zmq.h>
#include <memory>
#include <functional>
#include <thread>
#include <string>
#include <atomic>
#include <unordered_map>
#include <unistd.h>
#include <mutex>
#include <vector>
#define ZMQ_RPC_FUN  (ZMQ_REP | 0x80)
#define ZMQ_RPC_CALL (ZMQ_REQ | 0x80)

namespace StackFlows {

class pzmq_data {
private:
    zmq_msg_t msg;

public:
    pzmq_data(/* args */)
    {
        zmq_msg_init(&msg);
    }
    std::shared_ptr<std::string> get_string()
    {
        auto len = zmq_msg_size(&msg);
        return std::make_shared<std::string>((const char *)zmq_msg_data(&msg), zmq_msg_size(&msg));
    }
    std::string string()
    {
        auto len = zmq_msg_size(&msg);
        return std::string((const char *)zmq_msg_data(&msg), zmq_msg_size(&msg));
    }
    void *data()
    {
        return zmq_msg_data(&msg);
    }
    size_t size()
    {
        return zmq_msg_size(&msg);
    }
    zmq_msg_t *get()
    {
        return &msg;
    }
    ~pzmq_data()
    {
        zmq_msg_close(&msg);
    }
};

class pzmq {
public:
    typedef std::function<std::string(pzmq *, const std::shared_ptr<pzmq_data> &)> rpc_callback_fun;
    typedef std::function<void(pzmq *, const std::shared_ptr<pzmq_data> &)> msg_callback_fun;

private:
    const int rpc_url_head_length = 6;
    std::string rpc_url_head_     = "ipc:///tmp/rpc.";
    void *zmq_ctx_;
    void *zmq_socket_;
    std::unordered_map<std::string, rpc_callback_fun> zmq_fun_;
    std::mutex zmq_fun_mtx_;
    std::atomic<bool> flage_;
    std::unique_ptr<std::thread> zmq_thread_;
    int mode_;
    std::string rpc_server_;
    std::string zmq_url_;
    int timeout_;

    bool is_bind()
    {
        if ((mode_ == ZMQ_PUB) || (mode_ == ZMQ_PULL) || (mode_ == ZMQ_RPC_FUN))
            return true;
        else
            return false;
    }

public:
    pzmq() : zmq_ctx_(NULL), zmq_socket_(NULL), flage_(true), timeout_(3000)
    {
    }
    pzmq(const std::string &server)
        : zmq_ctx_(NULL), zmq_socket_(NULL), rpc_server_(server), flage_(true), timeout_(3000)
    {
        if (server.find("://") != std::string::npos) {
            rpc_url_head_.clear();
        }
    }
    pzmq(const std::string &url, int mode, const msg_callback_fun &raw_call = nullptr)
        : zmq_ctx_(NULL), zmq_socket_(NULL), mode_(mode), flage_(true), timeout_(3000)
    {
        if ((url[0] != 'i') && (url[1] != 'p')) {
            rpc_url_head_.clear();
        }
        if (mode_ != ZMQ_RPC_FUN) creat(url, raw_call);
    }
    void set_timeout(int ms)
    {
        timeout_ = ms;
    }
    int get_timeout()
    {
        return timeout_;
    }
    std::string get_zmq_url()
    {
        if ((!flage_.load()) && is_bind() && (zmq_url_.find("tcp://") != std::string::npos) &&
            (zmq_url_.find(":*") != std::string::npos)) {
            std::vector<char> zmq_url(zmq_url_.length() + 8, 0);
            size_t addr_len = zmq_url.size();
            if (zmq_getsockopt(zmq_socket_, ZMQ_LAST_ENDPOINT, zmq_url.data(), &addr_len) != 0) {
                return zmq_url_;
            }
            zmq_url_ = std::string(zmq_url.data());
        }
        return zmq_url_;
    }
    std::string _rpc_list_action(pzmq *self, const std::shared_ptr<pzmq_data> &_None)
    {
        std::string action_list;
        action_list.reserve(128);
        action_list = "{\"actions\":[";
        for (auto i = zmq_fun_.begin();;) {
            action_list += "\"";
            action_list += i->first;
            action_list += "\"";
            if (++i == zmq_fun_.end()) {
                action_list += "]}";
                break;
            } else {
                action_list += ",";
            }
        }
        return action_list;
    }
    int register_rpc_action(const std::string &action, const rpc_callback_fun &raw_call)
    {
        int ret = 0;
        std::unique_lock<std::mutex> lock(zmq_fun_mtx_);
        if (zmq_fun_.find(action) != zmq_fun_.end()) {
            zmq_fun_[action] = raw_call;
            return ret;
        }
        if (zmq_fun_.empty()) {
            std::string url = rpc_url_head_ + rpc_server_;
            mode_           = ZMQ_RPC_FUN;
            zmq_fun_["list_action"] =
                std::bind(&pzmq::_rpc_list_action, this, std::placeholders::_1, std::placeholders::_2);
            ret = creat(url);
        }
        zmq_fun_[action] = raw_call;
        return ret;
    }
    void unregister_rpc_action(const std::string &action)
    {
        std::unique_lock<std::mutex> lock(zmq_fun_mtx_);
        if (zmq_fun_.find(action) != zmq_fun_.end()) {
            zmq_fun_.erase(action);
        }
    }
    int call_rpc_action(const std::string &action, const std::string &data, const msg_callback_fun &raw_call)
    {
        int ret;
        std::shared_ptr<pzmq_data> msg_ptr = std::make_shared<pzmq_data>();
        try {
            if (NULL == zmq_socket_) {
                if (rpc_server_.empty()) return -1;
                std::string url = rpc_url_head_ + rpc_server_;
                mode_           = ZMQ_RPC_CALL;
                ret             = creat(url);
                if (ret) {
                    throw ret;
                }
            }
            // requist
            {
                zmq_send(zmq_socket_, action.c_str(), action.length(), ZMQ_SNDMORE);
                zmq_send(zmq_socket_, data.c_str(), data.length(), 0);
            }
            // action
            {
                zmq_msg_recv(msg_ptr->get(), zmq_socket_, 0);
            }
            raw_call(this, msg_ptr);
        } catch (int e) {
            ret = e;
        }
        msg_ptr.reset();
        close_zmq();
        return ret;
    }
    int creat(const std::string &url, const msg_callback_fun &raw_call = nullptr)
    {
        zmq_url_ = url;
        do {
            zmq_ctx_ = zmq_ctx_new();
        } while (zmq_ctx_ == NULL);
        do {
            zmq_socket_ = zmq_socket(zmq_ctx_, mode_ & 0x3f);
        } while (zmq_socket_ == NULL);

        switch (mode_) {
            case ZMQ_PUB: {
                return creat_pub(url);
            } break;
            case ZMQ_SUB: {
                int reconnect_interval = 100;
                zmq_setsockopt(zmq_socket_, ZMQ_RECONNECT_IVL, &reconnect_interval, sizeof(reconnect_interval));

                int max_reconnect_interval = 1000;  // 5 seconds
                zmq_setsockopt(zmq_socket_, ZMQ_RECONNECT_IVL_MAX, &max_reconnect_interval,
                               sizeof(max_reconnect_interval));
                return subscriber_url(url, raw_call);
            } break;
            case ZMQ_PUSH: {
                int reconnect_interval = 100;
                zmq_setsockopt(zmq_socket_, ZMQ_RECONNECT_IVL, &reconnect_interval, sizeof(reconnect_interval));
                int max_reconnect_interval = 1000;  // 5 seconds
                zmq_setsockopt(zmq_socket_, ZMQ_RECONNECT_IVL_MAX, &max_reconnect_interval,
                               sizeof(max_reconnect_interval));
                return creat_push(url);
            } break;
            case ZMQ_PULL: {
                return creat_pull(url, raw_call);
            } break;
            case ZMQ_RPC_FUN: {
                return creat_rep(url, raw_call);
            } break;
            case ZMQ_RPC_CALL: {
                return creat_req(url);
            } break;
            default:
                break;
        }
        return 0;
    }
    int check_zmq_errno(int code)
    {
        if (code == -1) {
            // SLOGE("Error occurred: %s", zmq_strerror(errno));
            switch (errno) {
                case EAGAIN:  // In non-blocking mode, the sending buffer is full, so the message cannot be sent
                              // immediately.
                    break;
                case ENOTSUP:  // The requested operation is not supported.
                    break;
                case EFSM:  // Inconsistent state machine with the socket.
                    break;
                case ETERM:  // The ZeroMQ context has been terminated.
                    break;
                case ENOTSOCK:  // The provided socket is invalid.
                    break;
                case EINTR:  // The operation was interrupted by a signal.
                    break;
                case EFAULT:  // Invalid buffer or socket.
                    break;
                default:
                    break;
            }
        }
        return code;
    }
    int send_data(const std::string &raw)
    {
        return zmq_send(zmq_socket_, raw.c_str(), raw.length(), 0);
    }
    int send_data(const char *raw, int size)
    {
        return zmq_send(zmq_socket_, raw, size, 0);
    }
    inline int creat_pub(const std::string &url)
    {
        return zmq_bind(zmq_socket_, url.c_str());
    }
    inline int creat_push(const std::string &url)
    {
        zmq_setsockopt(zmq_socket_, ZMQ_SNDTIMEO, &timeout_, sizeof(timeout_));
        return zmq_connect(zmq_socket_, url.c_str());
    }
    inline int creat_pull(const std::string &url, const msg_callback_fun &raw_call)
    {
        int ret     = zmq_bind(zmq_socket_, url.c_str());
        flage_      = false;
        zmq_thread_ = std::make_unique<std::thread>(std::bind(&pzmq::zmq_event_loop, this, raw_call));
        return ret;
    }
    inline int subscriber_url(const std::string &url, const msg_callback_fun &raw_call)
    {
        int ret = zmq_connect(zmq_socket_, url.c_str());
        zmq_setsockopt(zmq_socket_, ZMQ_SUBSCRIBE, "", 0);
        flage_      = false;
        zmq_thread_ = std::make_unique<std::thread>(std::bind(&pzmq::zmq_event_loop, this, raw_call));
        return ret;
    }
    inline int creat_rep(const std::string &url, const msg_callback_fun &raw_call)
    {
        int ret     = zmq_bind(zmq_socket_, url.c_str());
        flage_      = false;
        zmq_thread_ = std::make_unique<std::thread>(std::bind(&pzmq::zmq_event_loop, this, raw_call));
        return ret;
    }
    inline int creat_req(const std::string &url)
    {
        if (!rpc_url_head_.empty()) {
            std::string socket_file = url.substr(rpc_url_head_length);
            if (access(socket_file.c_str(), F_OK) != 0) {
                return -1;
            }
        }
        zmq_setsockopt(zmq_socket_, ZMQ_SNDTIMEO, &timeout_, sizeof(timeout_));
        zmq_setsockopt(zmq_socket_, ZMQ_RCVTIMEO, &timeout_, sizeof(timeout_));
        return zmq_connect(zmq_socket_, url.c_str());
    }
    void zmq_event_loop(const msg_callback_fun &raw_call)
    {
        int ret;
        zmq_pollitem_t items[1];
        if (mode_ == ZMQ_PULL) {
            items[0].socket  = zmq_socket_;
            items[0].fd      = 0;
            items[0].events  = ZMQ_POLLIN;
            items[0].revents = 0;
        };
        while (!flage_.load()) {
            std::shared_ptr<pzmq_data> msg_ptr = std::make_shared<pzmq_data>();
            if (mode_ == ZMQ_PULL) {
                ret = zmq_poll(items, 1, -1);
                if (ret == -1) {
                    zmq_close(zmq_socket_);
                    continue;
                }
                if (!(items[0].revents & ZMQ_POLLIN)) {
                    continue;
                }
            }
            ret = zmq_msg_recv(msg_ptr->get(), zmq_socket_, 0);
            if (ret <= 0) {
                msg_ptr.reset();
                continue;
            }

            if (mode_ == ZMQ_RPC_FUN) {
                std::shared_ptr<pzmq_data> msg1_ptr = std::make_shared<pzmq_data>();
                zmq_msg_recv(msg1_ptr->get(), zmq_socket_, 0);
                std::string retval;
                try {
                    std::unique_lock<std::mutex> lock(zmq_fun_mtx_);
                    retval = zmq_fun_.at(msg_ptr->string())(this, msg1_ptr);
                } catch (...) {
                    retval = "NotAction";
                }
                zmq_send(zmq_socket_, retval.c_str(), retval.length(), 0);
                msg1_ptr.reset();
            } else {
                raw_call(this, msg_ptr);
            }
            msg_ptr.reset();
        }
    }
    void close_zmq()
    {
        zmq_close(zmq_socket_);
        zmq_ctx_destroy(zmq_ctx_);
        if ((mode_ == ZMQ_PUB) || (mode_ == ZMQ_PULL) || (mode_ == ZMQ_RPC_FUN)) {
            if (!rpc_url_head_.empty()) {
                std::string socket_file = zmq_url_.substr(rpc_url_head_length);
                if (access(socket_file.c_str(), F_OK) == 0) {
                    remove(socket_file.c_str());
                }
            }
        }
        zmq_socket_ = NULL;
        zmq_ctx_    = NULL;
    }
    ~pzmq()
    {
        if (!zmq_socket_) {
            return;
        }
        flage_ = true;
        zmq_ctx_shutdown(zmq_ctx_);
        if (zmq_thread_) {
            zmq_thread_->join();
        }
        close_zmq();
    }

private:
    std::shared_ptr<void> context_ptr_;
    std::weak_ptr<void> wcontext_ptr_;
    void *ctx_;

public:
    // context
    void *context()
    {
        return ctx_;
    }
    void setContext(void *ctx)
    {
        ctx_ = ctx;
    }
    template <class T>
    T *newContext()
    {
        ctx_ = new T;
        return (T *)ctx_;
    }
    template <class T>
    T *getContext()
    {
        return (T *)ctx_;
    }
    template <class T>
    void deleteContext()
    {
        if (ctx_) {
            delete (T *)ctx_;
            ctx_ = NULL;
        }
    }

    // contextPtr
    std::shared_ptr<void> contextPtr()
    {
        return context_ptr_;
    }
    void setContextPtr(const std::shared_ptr<void> &ctx)
    {
        context_ptr_ = ctx;
    }
    void setContextPtr(std::shared_ptr<void> &&ctx)
    {
        context_ptr_ = std::move(ctx);
    }
    template <class T>
    std::shared_ptr<T> newContextPtr()
    {
        context_ptr_ = std::make_shared<T>();
        return std::static_pointer_cast<T>(context_ptr_);
    }
    template <class T>
    std::shared_ptr<T> getContextPtr()
    {
        return std::static_pointer_cast<T>(context_ptr_);
    }
    void deleteContextPtr()
    {
        context_ptr_.reset();
    }

    // wcontextPtr
    std::shared_ptr<void> wcontextPtr()
    {
        return wcontext_ptr_.lock();
    }
    void wsetContextPtr(const std::shared_ptr<void> &ctx)
    {
        wcontext_ptr_ = ctx;
    }
    template <class T>
    std::shared_ptr<T> wgetContextPtr()
    {
        return std::static_pointer_cast<T>(wcontext_ptr_.lock());
    }
    void wdeleteContextPtr()
    {
        wcontext_ptr_.reset();
    }
};
};  // namespace StackFlows