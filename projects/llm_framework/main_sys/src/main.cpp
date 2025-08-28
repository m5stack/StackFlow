/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include <signal.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <iostream>

#include "all.h"
#include "event_loop.h"
#include "serial.h"
#include "zmq_bus.h"
#include "remote_action.h"
#include "remote_server.h"
#include "linux_uart/linux_uart.h"

#ifdef ENABLE_BACKWARD
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
#include "backward.h"
#endif

pthread_spinlock_t key_sql_lock;
std::unordered_map<std::string, Any> key_sql;
std::string zmq_s_format;
std::string zmq_c_format;
int main_exit_flage = 0;

static void __sigint(int iSigNo)
{
    printf("llm_sys will be exit!\n");
    main_exit_flage = 1;
}

void get_run_config()
{
    key_sql["config_serial_dev"]            = std::string("/dev/tty1");
    key_sql["config_serial_baud"]           = 115200;
    key_sql["config_serial_data_bits"]      = 8;
    key_sql["config_serial_stop_bits"]      = 1;
    key_sql["config_serial_parity"]         = 110;  //'n' 的 asscii 值
    key_sql["config_serial_buffer_len"]     = 128;
    key_sql["config_serial_zmq_port"]       = 5556;
    key_sql["config_retry_time"]            = 3000;
    key_sql["config_work_id"]               = 1000;
    key_sql["config_unit_call_timeout"]     = 5000;
    key_sql["config_zmq_s_format"]          = std::string("ipc:///tmp/llm/%i.sock");
    key_sql["config_zmq_c_format"]          = std::string("ipc:///tmp/llm/%i.sock");
    key_sql["config_lsmod_dir"]             = std::string("/opt/m5stack/data/models/");
    key_sql["config_model_d"]               = std::string("/opt/m5stack/data/models/");
    key_sql["config_base_mode_path"]        = std::string("/opt/m5stack/data/");
    key_sql["config_base_mode_config_path"] = std::string("/opt/m5stack/etc/");
    key_sql["config_tcp_server"]            = 10001;
    key_sql["config_tcp_zmq_port"]          = 5557;
    key_sql["config_enable_tcp"]            = 1;
    key_sql["config_zmq_min_port"]          = 5010;
    key_sql["config_zmq_max_port"]          = 5555;
    key_sql["config_sys_pcm_play"]          = std::string("0.1");
    key_sql["config_sys_pcm_cap"]           = std::string("0.0");
    key_sql["config_sys_pcm_channel"]       = std::string("2");
    key_sql["config_sys_pcm_rate"]          = std::string("16000");
    key_sql["config_sys_pcm_bit"]           = std::string("S16_LE");
    key_sql["config_sys_pcm_cap_channel"]   = std::string("ipc:///tmp/llm/pcm.cap.socket");
    key_sql["config_sys_stream_length"]     = 1024;
    load_default_config();
}

#define uart_reset_check_obj(obj, config) \
    if (obj != config) {                  \
        obj        = config;              \
        reset_init = 1;                   \
    }

void uart_reset_check()
{
    uart_t uart_parm;
    uart_parm.baud      = any_cast<int>(key_sql["config_serial_baud"]);
    uart_parm.data_bits = any_cast<int>(key_sql["config_serial_data_bits"]);
    uart_parm.stop_bits = any_cast<int>(key_sql["config_serial_stop_bits"]);
    uart_parm.parity    = any_cast<int>(key_sql["config_serial_parity"]);
    int reset_init      = 0;
    uart_reset_check_obj(uart_parm.baud, 115200);
    uart_reset_check_obj(uart_parm.data_bits, 8);
    uart_reset_check_obj(uart_parm.stop_bits, 1);
    uart_reset_check_obj(uart_parm.parity, 110);
    uart_parm.wait_flage = 1;
    if (reset_init) {
        const char *dev_name = "/dev/tty1";
        int uart_fd          = linux_uart_init((char *)dev_name, &uart_parm);
        char uart_buff[32];
        struct timespec _run_time_start;
        clock_gettime(CLOCK_MONOTONIC, &_run_time_start);
        const char *sendmsg = "llm uart reset\n";
        linux_uart_write(uart_fd, 15, (char *)sendmsg);
        const char *flage_str = "llm_uart_reset";
        char *flage_strp      = (char *)flage_str;
        while (1) {
            int len = linux_uart_read(uart_fd, 32, uart_buff);
            for (int i = 0; i < len; i++) {
                if (uart_buff[i] != *flage_strp++) {
                    flage_strp = (char *)flage_str;
                }
                if (*flage_strp == '\0') {
                    key_sql["config_serial_dev"]       = std::string("/dev/tty1");
                    key_sql["config_serial_baud"]      = 115200;
                    key_sql["config_serial_data_bits"] = 8;
                    key_sql["config_serial_stop_bits"] = 1;
                    key_sql["config_serial_parity"]    = 110;  //'n' 的 asscii 值
                    linux_uart_deinit(uart_fd);
                    break;
                }
            }
            struct timespec _run_time_end;
            clock_gettime(CLOCK_MONOTONIC, &_run_time_end);
            if (((float)(((_run_time_end.tv_sec * 1000000 + _run_time_end.tv_nsec / 1000) -
                          (_run_time_start.tv_sec * 1000000 + _run_time_start.tv_nsec / 1000)) /
                         1000.f)) > 1000) {
                linux_uart_deinit(uart_fd);
                break;
            }
        }
    }
}

void tcp_work();

void tcp_stop_work();

void all_work()
{
    zmq_s_format = any_cast<std::string>(key_sql["config_zmq_s_format"]);
    zmq_c_format = any_cast<std::string>(key_sql["config_zmq_c_format"]);
    server_work();
    remote_server_work();
    remote_action_work();
    zmq_bus_work();
    serial_work();
    int enable_tcp = 0;
    SAFE_READING(enable_tcp, int, "config_enable_tcp");
    if (enable_tcp) tcp_work();
}

void all_stop_work()
{
    int enable_tcp = 0;
    SAFE_READING(enable_tcp, int, "config_enable_tcp");

    if (enable_tcp) tcp_stop_work();
    serial_stop_work();
    remote_server_stop_work();
    remote_action_stop_work();
    zmq_bus_stop_work();
}

void all_work_check()
{
}

int main(int argc, char *argv[])
{
    signal(SIGTERM, __sigint);
    signal(SIGINT, __sigint);
    mkdir("/tmp/llm", 0777);
    if (pthread_spin_init(&key_sql_lock, PTHREAD_PROCESS_PRIVATE) != 0) {
        SLOGE("key_sql_lock init false");
        exit(1);
    }
    SLOGD("llm_sys start");
    get_run_config();
    uart_reset_check();
    all_work();
    SLOGD("llm_sys work");
    while (main_exit_flage == 0) {
        sleep(1);
        all_work_check();
    }
    SLOGD("llm_sys stop");
    all_stop_work();
    pthread_spin_destroy(&key_sql_lock);
    return 0;
}