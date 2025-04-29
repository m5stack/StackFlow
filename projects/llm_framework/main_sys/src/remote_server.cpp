
/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include <mutex>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include "all.h"
#include "remote_server.h"
#include "zmq_bus.h"
#include <cstring>
#include <StackFlowUtil.h>
using namespace StackFlows;

std::atomic<int> work_id_number_counter;
int port_list_start;
std::vector<bool> port_list;
std::unique_ptr<pzmq> sys_rpc_server_;

std::string sys_sql_select(const std::string &key)
{
    std::string out;
    SAFE_READING(out, std::string, key);
    return out;
}

void sys_sql_set(const std::string &key, const std::string &val)
{
    SAFE_SETTING(key, val);
}

void sys_sql_unset(const std::string &key)
{
    SAFE_ERASE(key);
}

unit_data *sys_allocate_unit(const std::string &unit)
{
    unit_data *unit_p = new unit_data();
    {
        unit_p->port_     = work_id_number_counter++;
        std::string ports = std::to_string(unit_p->port_);
        unit_p->work_id   = unit + "." + ports;
    }
    {
        int port;
        for (size_t i = 0; i < port_list.size(); i++) {
            if (!port_list[i]) {
                port         = port_list_start + i;
                port_list[i] = true;
                break;
            }
        }
        std::string ports      = std::to_string(port);
        std::string zmq_format = zmq_s_format;
        if (zmq_s_format.find("sock") != std::string::npos) {
            zmq_format += ".";
            zmq_format += unit;
            zmq_format += ".output_url";
        }
        std::vector<char> buff(zmq_format.length() + ports.length(), 0);
        sprintf((char *)buff.data(), zmq_format.c_str(), port);
        std::string zmq_s_url = std::string((char *)buff.data());
        unit_p->output_url    = zmq_s_url;
    }
    {
        int port;
        for (size_t i = 0; i < port_list.size(); i++) {
            if (!port_list[i]) {
                port         = port_list_start + i;
                port_list[i] = true;
                break;
            }
        }
        std::string ports      = std::to_string(port);
        std::string zmq_format = zmq_c_format;
        if (zmq_s_format.find("sock") != std::string::npos) {
            zmq_format += ".";
            zmq_format += unit;
            zmq_format += ".input_url";
        }
        std::vector<char> buff(zmq_format.length() + ports.length(), 0);
        sprintf((char *)buff.data(), zmq_format.c_str(), port);
        std::string zmq_c_url = std::string((char *)buff.data());
        unit_p->input_url     = zmq_c_url;
    }
    {
        int port;
        for (size_t i = 0; i < port_list.size(); i++) {
            if (!port_list[i]) {
                port         = port_list_start + i;
                port_list[i] = true;
                break;
            }
        }
        std::string ports      = std::to_string(port);
        std::string zmq_format = zmq_s_format;
        if (zmq_s_format.find("sock") != std::string::npos) {
            zmq_format += ".";
            zmq_format += unit;
            zmq_format += ".inference_url";
        }
        std::vector<char> buff(zmq_format.length() + ports.length(), 0);
        sprintf((char *)buff.data(), zmq_format.c_str(), port);
        std::string zmq_s_url = std::string((char *)buff.data());
        unit_p->init_zmq(zmq_s_url);
    }
    SAFE_SETTING(unit_p->work_id, unit_p);
    SAFE_SETTING(unit_p->work_id + ".out_port", unit_p->output_url);
    return unit_p;
}

int sys_release_unit(const std::string &unit)
{
    unit_data *unit_p = NULL;
    SAFE_READING(unit_p, unit_data *, unit);
    if (NULL == unit_p) {
        return -1;
    }

    int port;
    sscanf(unit_p->output_url.c_str(), zmq_s_format.c_str(), &port);
    port_list[port - port_list_start] = false;
    sscanf(unit_p->input_url.c_str(), zmq_c_format.c_str(), &port);
    port_list[port - port_list_start] = false;
    sscanf(unit_p->inference_url.c_str(), zmq_s_format.c_str(), &port);
    port_list[port - port_list_start] = false;

    delete unit_p;
    SAFE_ERASE(unit);
    SAFE_ERASE(unit + ".out_port");
    return 0;
}

char const *c_sys_sql_select(char const *key)
{
    static std::string val;
    val = sys_sql_select(key);
    return val.c_str();
}

void c_sys_sql_set(char const *key, char const *val)
{
    sys_sql_set(key, val);
}

void c_sys_sql_unset(char const *key)
{
    sys_sql_unset(key);
}

void c_sys_allocate_unit(char const *unit, char *input_url, char *output_url, int *work_id)
{
    unit_data *unit_p = sys_allocate_unit(unit);
    strcpy(input_url, unit_p->inference_url.c_str());
    strcpy(output_url, unit_p->output_url.c_str());

    std::string work_id_format = unit;
    work_id_format += ".%i";
    sscanf(unit_p->work_id.c_str(), work_id_format.c_str(), work_id);
}

int c_sys_release_unit(char const *unit)
{
    return sys_release_unit(unit);
}

std::string rpc_allocate_unit(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw)
{
    unit_data *unit_info = sys_allocate_unit(raw->string());
    std::string send_data;
    std::string send_data1;
    std::string str_port = std::to_string(unit_info->port_);
    RPC_PUSH_PARAM(send_data1, unit_info->output_url, unit_info->inference_url);
    RPC_PUSH_PARAM(send_data, str_port, send_data1);
    return send_data;
}

std::string rpc_release_unit(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw)
{
    sys_release_unit(raw->string());
    return "Success";
}

std::string rpc_sql_select(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw)
{
    return sys_sql_select(raw->string());
}

std::string rpc_sql_set(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw)
{
    std::string key = sample_json_str_get(raw->string(), "key");
    std::string val = sample_json_str_get(raw->string(), "val");
    if (key.empty()) return "False";
    sys_sql_set(key, val);
    return "Success";
}

std::string rpc_sql_unset(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw)
{
    sys_sql_unset(raw->string());
    return "Success";
}

void remote_server_work()
{
    int port_list_end;
    SAFE_READING(work_id_number_counter, int, "config_work_id");
    SAFE_READING(port_list_start, int, "config_zmq_min_port");
    SAFE_READING(port_list_end, int, "config_zmq_max_port");
    port_list.resize(port_list_end - port_list_start, 0);

    sys_rpc_server_ = std::make_unique<pzmq>("sys");
    sys_rpc_server_->register_rpc_action("sql_select",
                                         std::bind(rpc_sql_select, std::placeholders::_1, std::placeholders::_2));
    sys_rpc_server_->register_rpc_action("register_unit",
                                         std::bind(rpc_allocate_unit, std::placeholders::_1, std::placeholders::_2));
    sys_rpc_server_->register_rpc_action("release_unit",
                                         std::bind(rpc_release_unit, std::placeholders::_1, std::placeholders::_2));
    sys_rpc_server_->register_rpc_action("sql_set",
                                         std::bind(rpc_sql_set, std::placeholders::_1, std::placeholders::_2));
    sys_rpc_server_->register_rpc_action("sql_unset",
                                         std::bind(rpc_sql_unset, std::placeholders::_1, std::placeholders::_2));
}

void remote_server_stop_work()
{
    sys_rpc_server_.reset();
}