/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "event_loop.h"

#include <fcntl.h>
#include <pty.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <thread>
#include <fstream>

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <list>
#include <map>
#include <vector>

#include "all.h"
#include "json.hpp"
#include "serial.h"
#include "subprocess.h"
#include "zmq_bus.h"
#include "remote_action.h"
#include <simdjson.h>
#include "hv/ifconfig.h"
#include <glob.h>
#include "StackFlowUtil.h"

void usr_print_error(const std::string &request_id, const std::string &work_id, const std::string &error_msg,
                     int zmq_out)
{
    nlohmann::json out_body;
    out_body["request_id"] = request_id;
    out_body["work_id"]    = work_id;
    out_body["created"]    = time(NULL);
    out_body["error"]      = nlohmann::json::parse(error_msg);
    out_body["object"]     = std::string("None");
    out_body["data"]       = std::string("None");
    std::string out        = out_body.dump();
    zmq_com_send(zmq_out, out);
}

template <typename T>
void usr_out(const std::string &request_id, const std::string &work_id, const T &data, int zmq_out)
{
    nlohmann::json out_body;
    out_body["request_id"] = request_id;
    out_body["work_id"]    = work_id;
    out_body["created"]    = time(NULL);
    if constexpr (std::is_same<T, std::string>::value) {
        out_body["object"] = std::string("sys.utf-8");
    } else {
        out_body["object"] = std::string("sys.utf-8.stream");
    }
    out_body["data"]  = data;
    out_body["error"] = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");
    std::string out   = out_body.dump();
    zmq_com_send(zmq_out, out);
}

int sys_ping(int com_id, const nlohmann::json &json_obj)
{
    int out = 0;
    usr_print_error(json_obj["request_id"], json_obj["work_id"], "{\"code\":0, \"message\":\"\"}", com_id);
    return out;
}

void _sys_uartsetup(int com_id, const nlohmann::json &json_obj)
{
    SAFE_SETTING("config_serial_baud", (int)json_obj["data"]["baud"]);
    SAFE_SETTING("config_serial_data_bits", (int)json_obj["data"]["data_bits"]);
    SAFE_SETTING("config_serial_stop_bits", (int)json_obj["data"]["stop_bits"]);
    SAFE_SETTING("config_serial_parity", (int)json_obj["data"]["parity"]);
    usr_print_error(json_obj["request_id"], json_obj["work_id"], "{\"code\":0, \"message\":\"\"}", com_id);
    usleep(100 * 1000);
    serial_stop_work();
    serial_work();
}

int sys_uartsetup(int com_id, const nlohmann::json &json_obj)
{
    int out = 0;
    std::thread t(_sys_uartsetup, com_id, json_obj);
    t.detach();
    return out;
}

void get_memory_info(unsigned long *total_memory, unsigned long *free_memory)
{
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo == NULL) {
        perror("fopen");
        return;
    }
    char line[256];
    while (fgets(line, sizeof(line), meminfo)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line, "MemTotal: %lu kB", total_memory);
        }
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line, "MemAvailable: %lu kB", free_memory);
            break;
        }
    }
    fclose(meminfo);
}

struct cpu_use_t {
    long double a[4];
    long time;
};

int _sys_hwinfo(int com_id, const nlohmann::json &json_obj)
{
    unsigned long one;
    unsigned long two;
    unsigned long temp;
    struct cpu_use_t now;
    struct cpu_use_t laster;
    long double loadavg;
    FILE *fp;
    nlohmann::json out_body;
    nlohmann::json data_body;
    out_body["request_id"] = json_obj["request_id"];
    out_body["work_id"]    = std::string("sys");
    out_body["created"]    = time(NULL);
    out_body["error"]      = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");
    out_body["object"]     = std::string("sys.hwinfo");

    fp = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    fscanf(fp, "%Li", &temp);
    fclose(fp);
    data_body["temperature"] = temp;

    fp = fopen("/proc/stat", "r");
    fscanf(fp, "%*s %Lf %Lf %Lf %Lf", &laster.a[0], &laster.a[1], &laster.a[2], &laster.a[3]);
    fclose(fp);
    sleep(1);
    fp = fopen("/proc/stat", "r");
    fscanf(fp, "%*s %Lf %Lf %Lf %Lf", &now.a[0], &now.a[1], &now.a[2], &now.a[3]);
    fclose(fp);
    loadavg = ((now.a[0] + now.a[1] + now.a[2]) - (laster.a[0] + laster.a[1] + laster.a[2])) /
              ((now.a[0] + now.a[1] + now.a[2] + now.a[3]) - (laster.a[0] + laster.a[1] + laster.a[2] + laster.a[3]));
    data_body["cpu_loadavg"] = (int)(loadavg * 100);

    get_memory_info(&one, &two);
    float use        = ((one * 1.0f) - (two * 1.0f)) / (one * 1.0f);
    data_body["mem"] = (int)(use * 100);
    std::vector<ifconfig_t> ifcs;
    ifconfig(ifcs);
    std::vector<nlohmann::json> jifcs;
    for (auto &item : ifcs) {
        nlohmann::json eth_info;
        eth_info["name"]      = item.name;
        eth_info["ip"]        = item.ip;
        char eth_ip_buff[128] = {0};
        sprintf(eth_ip_buff, "/sys/class/net/%s/speed", item.name);
        FILE *file = fopen(eth_ip_buff, "r");
        memset(eth_ip_buff, 0, sizeof(eth_ip_buff));
        if (file != NULL) {
            int size = fread(eth_ip_buff, 1, sizeof(eth_ip_buff), file);
            if (size > 0) eth_ip_buff[size - 1] = '\0';
            fclose(file);
        }
        eth_info["speed"] = eth_ip_buff;
        jifcs.push_back(eth_info);
    }
    data_body["eth_info"] = jifcs;
    out_body["data"]      = data_body;
    std::string out       = out_body.dump();
    zmq_com_send(com_id, out);
    return 0;
}

int sys_hwinfo(int com_id, const nlohmann::json &json_obj)
{
    int out = 0;
    std::thread t(_sys_hwinfo, com_id, json_obj);
    t.detach();
    return out;
}

int _sys_unit_call(int com_id, const nlohmann::json &json_obj)
{
    std::string json_obj_raw = json_obj.dump();
    std::string object       = sample_json_str_get(json_obj_raw, "object");
    std::string data         = sample_json_str_get(json_obj_raw, "data");
    std::string out          = unit_call(object.substr(0, object.find(".")), object.substr(object.find(".") + 1), data);
    {
        nlohmann::json out_body;
        out_body["request_id"] = json_obj["request_id"];
        out_body["work_id"]    = json_obj["work_id"];
        out_body["created"]    = time(NULL);
        out_body["object"]     = json_obj["object"];
        try {
            out_body["data"] = nlohmann::json::parse(out);
        } catch (...) {
            out_body["data"] = out;
        }
        out_body["error"] = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");
        zmq_com_send(com_id, out_body.dump());
    }
    return 0;
}

int sys_unit_call(int com_id, const nlohmann::json &json_obj)
{
    int out = 0;
    std::thread t(_sys_unit_call, com_id, json_obj);
    t.detach();
    return out;
}

void get_mem_cmm_info(unsigned long *total_size, unsigned long *used, unsigned long *remain)
{
    std::ifstream file("/proc/ax_proc/mem_cmm_info");
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    if (!lines.empty()) {
        std::string last_line = lines.back();

        size_t pos = last_line.find("total size=");
        if (pos != std::string::npos) {
            pos += 11;
            size_t end  = last_line.find('K', pos);
            *total_size = std::stoul(last_line.substr(pos, end - pos));
        }

        pos = last_line.find("used=");
        if (pos != std::string::npos) {
            pos += 5;
            size_t end = last_line.find('K', pos);
            *used      = std::stoul(last_line.substr(pos, end - pos));
        }

        pos = last_line.find("remain=");
        if (pos != std::string::npos) {
            pos += 7;
            size_t end = last_line.find('K', pos);
            *remain    = std::stoul(last_line.substr(pos, end - pos));
        }
    }
}

void _sys_cmminfo(int com_id, const nlohmann::json &json_obj)
{
    unsigned long total_size, used, remain;
    get_mem_cmm_info(&total_size, &used, &remain);

    nlohmann::json out_body;
    nlohmann::json data_body;
    out_body["request_id"] = json_obj["request_id"];
    out_body["work_id"]    = std::string("sys");
    out_body["created"]    = time(NULL);
    out_body["error"]      = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");
    out_body["object"]     = std::string("sys.cmminfo");
    data_body["total"]     = total_size;
    data_body["used"]      = used;
    data_body["remain"]    = remain;
    out_body["data"]       = data_body;
    std::string out        = out_body.dump();
    zmq_com_send(com_id, out);
}

int sys_cmminfo(int com_id, const nlohmann::json &json_obj)
{
    int out = 0;
    std::thread t(_sys_cmminfo, com_id, json_obj);
    t.detach();
    return out;
}

int sys_lsmode(int com_id, const nlohmann::json &json_obj)
{
    int out;
    nlohmann::json out_body;
    int stream = false;
    DIR *dir;
    struct dirent *entry;
    std::list<nlohmann::json> mode_body_list;

    out_body["request_id"] = json_obj["request_id"];
    out_body["work_id"]    = json_obj["work_id"];
    out_body["created"]    = time(NULL);
    out_body["object"]     = "sys.lsmode";
    out_body["data"]       = mode_body_list;
    out_body["error"]      = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");

    std::string lsmode_dir;
    SAFE_READING(lsmode_dir, std::string, "config_lsmod_dir");
    dir = opendir(lsmode_dir.c_str());

    if (dir == NULL) {
        perror("opendir");
        goto sys_lsmode_err_1;
    }

    if (stream) {
    } else {
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".json") != NULL) {
                int file_size;
                FILE *file = fopen((lsmode_dir + entry->d_name).c_str(), "r");
                fseek(file, 0L, SEEK_END);
                file_size = ftell(file);
                fseek(file, 0L, SEEK_SET);

                std::vector<char> file_buffer(file_size + 1);

                int size = fread(file_buffer.data(), 1, file_buffer.size(), file);
                try {
                    nlohmann::json mode_body = nlohmann::json::parse(std::string(file_buffer.data(), size));
                    mode_body_list.push_back(mode_body);
                } catch (const nlohmann::json::parse_error &e) {
                    SLOGW("%s json format error", (lsmode_dir + entry->d_name).c_str());
                }
                fclose(file);
            }
        }
        out_body["created"] = time(NULL);
        out_body["object"]  = "sys.lsmode";
        out_body["data"]    = mode_body_list;
        out_body["error"]   = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");
    }

    closedir(dir);

sys_lsmode_err_1:
    zmq_com_send(com_id, out_body.dump());
    return out;
}

int sys_rmmode(int com_id, const nlohmann::json &json_obj)
{
    std::string rmmode_name   = json_obj["data"];
    const std::string command = "dpkg -P llm-" + rmmode_name;
    int out                   = system(command.c_str());
    usr_print_error(json_obj["request_id"], json_obj["work_id"], "{\"code\":0, \"message\":\"\"}", com_id);
    return out;
}

int sys_lstask(int com_id, const nlohmann::json &json_obj)
{
    int out;

sys_lstask_err_1:
    usr_print_error(json_obj["request_id"], json_obj["work_id"],
                    "{\"code\":-10, \"message\":\"Not available at the moment.\"}", com_id);
    return out;
}

bool read_file(const std::string &path, std::vector<char> &data)
{
    std::fstream fs(path, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        return false;
    }
    fs.seekg(std::ios::end);
    auto fs_end = fs.tellg();
    fs.seekg(std::ios::beg);
    auto fs_beg      = fs.tellg();
    auto file_size   = static_cast<size_t>(fs_end - fs_beg);
    auto vector_size = data.size();
    data.reserve(vector_size + file_size);
    data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
    fs.close();
    return true;
}

bool dump_file(const std::string &path, char *data, int size)
{
    std::fstream fs(path, std::ios::out | std::ios::binary);
    if (!fs.is_open() || fs.fail()) {
        fprintf(stderr, "[ERR] cannot open file %s \n", path.c_str());
    }
    fs.write(data, size);
    return true;
}

// sys.file./opt/data/test.txt
// sys.stream.file./opt/data/test.txt
// sys.base64.file./opt/data/test.bin
// sys.base64.stream.file./opt/data.bin
int sys_push(int com_id, const nlohmann::json &json_obj)
{
    int out               = 0;
    std::string object    = json_obj["object"];
    bool stream           = (object.find("stream") != std::string::npos);
    bool base64           = (object.find("base64") != std::string::npos);
    std::string file_path = object.substr(object.find("file") + 5);
    if (file_path[0] != '/') {
        usr_print_error(json_obj["request_id"], json_obj["work_id"], "{\"code\":-17, \"message\":\"file path error\"}",
                        com_id);
        return out;
    }
    std::string file_dir = file_path.substr(0, file_path.rfind("/") + 1);
    file_dir += ".tmp_file";
    if (access(file_dir.c_str(), F_OK) != 0) {
        mkdir(file_dir.c_str(), 0777);
    }
    std::string file_panduan;
    if (stream) {
        int index         = json_obj["data"]["index"];
        bool finish       = json_obj["data"]["finish"];
        std::string delta = json_obj["data"]["delta"];
        file_panduan      = file_dir + std::string("/") + std::to_string(index) + std::string(".bin");
        dump_file(file_panduan, (char *)delta.c_str(), delta.length());
        if (!finish) {
            return out;
        }
    } else {
        file_panduan     = file_dir + std::string("/") + std::string("0.bin");
        std::string data = json_obj["data"];
        dump_file(file_panduan, (char *)data.c_str(), data.length());
    }
    std::string merah_cmd;
    if (base64) {
        merah_cmd = R"(find )";
        merah_cmd += file_dir;
        merah_cmd += R"( -maxdepth 1 -name "*.bin" | sort -n | xargs -i cat {} >> )";
        merah_cmd += file_dir + "/src.base64";
        // std::cout << "merah_cmd: " << merah_cmd << "\n" ;
        system(merah_cmd.c_str());
        merah_cmd = R"(base64 -d )";
        merah_cmd += file_dir + "/src.base64";
        merah_cmd += R"( > )";
        merah_cmd += file_path;
        // std::cout << "merah_cmd: " << merah_cmd << "\n" ;
        system(merah_cmd.c_str());
    } else {
        merah_cmd = R"(rm -f )";
        merah_cmd += file_path;
        system(merah_cmd.c_str());
        merah_cmd = R"(find )";
        merah_cmd += file_dir;
        merah_cmd += R"( -maxdepth 1 -name "*.bin" | sort -n | xargs -i cat {} >> )";
        merah_cmd += file_path;
        system(merah_cmd.c_str());
    }
    merah_cmd = R"(rm -rf )";
    merah_cmd += file_dir;
    system(merah_cmd.c_str());
    merah_cmd = R"(sha256sum )";
    merah_cmd += file_path;
    merah_cmd += "> /tmp";
    merah_cmd += file_path.substr(file_path.rfind("/") + 1);
    merah_cmd += ".base64_info";
    system(merah_cmd.c_str());
    merah_cmd = "/tmp";
    merah_cmd += file_path.substr(file_path.rfind("/") + 1);
    merah_cmd += ".base64_info";
    std::vector<char> base64_info;
    read_file(merah_cmd, base64_info);
    merah_cmd = std::string("rm -rf ") + merah_cmd;
    system(merah_cmd.c_str());
    std::string base64_info_str = std::string(base64_info.data(), 64);
    std::string error_str       = "{\"code\":0, \"message\":\"sha256:";
    error_str += base64_info_str;
    error_str += "\"}";
    usr_print_error(json_obj["request_id"], json_obj["work_id"], error_str, com_id);
    return out;
}

int sys_pull(int com_id, const nlohmann::json &json_obj)
{
    int out               = 0;
    std::string object    = json_obj["object"];
    bool stream           = (object.find("stream") != std::string::npos);
    std::string file_path = object.substr(object.find("file") + 5);
    if (access(file_path.c_str(), F_OK) != 0) {
        usr_print_error(json_obj["request_id"], json_obj["work_id"],
                        "{\"code\":-17, \"message\":\"file does not exist.\"}", com_id);
        return -1;
    }
    std::vector<char> file_data;
    read_file(file_path, file_data);
    std::string base64_data;
    StackFlows::encode_base64(std::string(file_data.data(), file_data.size()), base64_data);
    if (stream) {
        int config_sys_stream_length;
        SAFE_READING(config_sys_stream_length, int, "config_sys_stream_length");
        int send_pos = 0;
        nlohmann::json out_body;
        out_body["request_id"] = json_obj["request_id"];
        out_body["work_id"]    = "sys";
        out_body["created"]    = time(NULL);
        out_body["object"]     = std::string("sys.base64.stream");
        out_body["error"]      = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");
        size_t i               = 0;
        for (;;) {
            out_body["data"]["index"]  = i;
            out_body["data"]["finish"] = false;
            if ((send_pos + config_sys_stream_length) < base64_data.size()) {
                out_body["data"]["delta"] = base64_data.substr(send_pos, config_sys_stream_length);
                send_pos += config_sys_stream_length;
                zmq_com_send(com_id, out_body.dump());
            } else {
                out_body["data"]["delta"] = base64_data.substr(send_pos);
                zmq_com_send(com_id, out_body.dump());
                i++;
                break;
            }
            i++;
        };
        out_body["data"]["index"]  = i;
        out_body["data"]["finish"] = true;
        out_body["data"]["delta"]  = std::string("");
        zmq_com_send(com_id, out_body.dump());
    } else {
        nlohmann::json out_body;
        out_body["request_id"] = json_obj["request_id"];
        out_body["work_id"]    = "sys";
        out_body["created"]    = time(NULL);
        out_body["object"]     = std::string("sys.base64.stream");
        out_body["error"]      = nlohmann::json::parse("{\"code\":0, \"message\":\"\"}");
        out_body["data"]       = base64_data;
        zmq_com_send(com_id, out_body.dump());
    }
    return out;
}

int sys_update(int com_id, const nlohmann::json &json_obj)
{
    int out;
    const std::string command = "find /mnt -name \"llm_update_*.deb\" >> /tmp/find_update_file_out.txt 2>&1 ";
    out                       = system(command.c_str());
    std::ifstream file("/tmp/find_update_file_out.txt");
    if (!file.is_open()) {
        usr_print_error(json_obj["request_id"], json_obj["work_id"],
                        "{\"code\":-17, \"message\":\"Internal error, file opening failed.\"}", com_id);
        return -1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string lines = buffer.str();
    usr_out(json_obj["request_id"], json_obj["work_id"], lines, com_id);
    file.close();
    std::remove("/tmp/find_update_file_out.txt");
    return out;

sys_update_err_1:
    usr_print_error(json_obj["request_id"], json_obj["work_id"],
                    "{\"code\":-10, \"message\":\"Not available at the moment.\"}", com_id);
    return out;
}

int sys_upgrade(int com_id, const nlohmann::json &json_obj)
{
    int out;
    std::string version;
    try {
        version = json_obj["data"];
    } catch (...) {
        usr_print_error(json_obj["request_id"], json_obj["work_id"],
                        "{\"code\":-2, \"message\":\"json sys.version get false.\"}", com_id);
        return -1;
    }
    usr_out(json_obj["request_id"], json_obj["work_id"], std::string("update ..."), com_id);
    const std::string command =
        "bash -c 'touch /var/llm_update.lock ; find /mnt -name \"%s\" -print0 | xargs -0 -I {} dpkg -i {} ; ' > "
        "/var/llm_update.log 2>&1 & ";
    std::vector<char> buff(command.length() + version.length() + 8, 0);
    sprintf(buff.data(), command.c_str(), version.c_str());
    system((const char *)buff.data());
    return out;

sys_upgrade_err_1:
    usr_print_error(json_obj["request_id"], json_obj["work_id"],
                    "{\"code\":-10, \"message\":\"Not available at the moment.\"}", com_id);
    return out;
}

int _sys_bashexec(int com_id, std::string request_id, std::string work_id, std::string bashcmd, int stream, int base64)
{
    int out;
    int stream_size = 512;
    /****************************************************/
    int master_fd;
    int slave_fd;
    pid_t pid;
    ssize_t nread;

    pid = forkpty(&master_fd, nullptr, nullptr, nullptr);

    if (pid == -1) {
        perror("forkpty");
        goto sys_bashexec_err_1;
    } else if (pid == 0) {
        setenv("PS1", "", 1);
        unsetenv("TERM");
        dup2(STDOUT_FILENO, STDERR_FILENO);
        execl("/bin/bash", "bash", nullptr);
        perror("execl");
        exit(EXIT_FAILURE);
    } else {
        struct termios tty;
        tcgetattr(master_fd, &tty);
        tty.c_lflag &= ~ECHO;
        tcsetattr(master_fd, TCSANOW, &tty);
        bashcmd += " \r exit \r ";
        write(master_fd, bashcmd.c_str(), bashcmd.length());
        if (stream) {
            int index = 0;
            std::vector<char> _buffer(stream_size);
            char *buffer = _buffer.data();
            while ((nread = read(master_fd, buffer, stream_size - 1)) > 0) {
                buffer[nread] = '\0';
                nlohmann::json delta_body;
                delta_body["index"]  = index++;
                delta_body["delta"]  = std::string(buffer);
                delta_body["finish"] = false;
                usr_out(request_id, work_id, delta_body, com_id);
            }
            nlohmann::json delta_body;
            delta_body["index"]  = index++;
            delta_body["delta"]  = std::string("");
            delta_body["finish"] = true;
            usr_out(request_id, work_id, delta_body, com_id);
        } else {
            std::string process_out;
            char process_out_buffer[512];
            while ((nread = read(master_fd, process_out_buffer, sizeof(process_out_buffer) - 1)) > 0) {
                process_out_buffer[nread] = '\0';
                process_out += process_out_buffer;
            }
            usr_out(request_id, work_id, process_out, com_id);
        }
        kill(pid, SIGKILL);
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            std::cout << "Child process exited with status " << WEXITSTATUS(status) << std::endl;
        }
    }
    close(master_fd);
    /****************************************************/
sys_bashexec_err_1:
    return out;
}

int sys_bashexec(int com_id, const nlohmann::json &json_obj)
{
    std::string request_id = json_obj["request_id"];
    std::string work_id    = json_obj["work_id"];
    std::string bashcmd;
    std::string object = json_obj["object"];

    int stream = 0;
    int base64 = 0;
    if (object.find("stream") != std::string::npos) stream = 1;
    if (object.find("base64") != std::string::npos) base64 = 1;
    static std::map<int, std::string> bash_delta;
    if (stream) {
        bash_delta[(int)json_obj["data"]["index"]] = (std::string)json_obj["data"]["delta"];

        usr_print_error(request_id, work_id, "{\"code\":0, \"message\":\"\"}", com_id);
        if (!json_obj["data"]["finish"]) {
            return 0;
        } else {
            for (size_t i = 0; i < bash_delta.size(); i++) {
                bashcmd += bash_delta[i];
            }
            bash_delta.clear();
        }
    } else {
        bashcmd = (std::string)json_obj["data"];
    }
    if (base64) {
    } else {
    }
    std::thread t(_sys_bashexec, com_id, request_id, work_id, bashcmd, stream, base64);
    t.detach();
    return 0;
}

int sys_reset(int com_id, const nlohmann::json &json_obj)
{
    int out = 0;
    usr_print_error(json_obj["request_id"], json_obj["work_id"],
                    "{\"code\":0, \"message\":\"llm server restarting ...\"}", com_id);
    const char *cmd =
        "[ -f '/tmp/llm/reset.lock' ] || bash -c \"touch /tmp/llm_reset.lock ; sync ; "
        "systemctl restart llm-* \" > /dev/null 2>&1 & ";
    system(cmd);
    return out;
}

int sys_version(int com_id, const nlohmann::json &json_obj)
{
    usr_out(json_obj["request_id"], json_obj["work_id"], std::string("v1.6"), com_id);

    int out = 0;
    return out;
}

int sys_version2(int com_id, const nlohmann::json &json_obj)
{
    nlohmann::json data_body = nlohmann::json::array();
    glob_t glob_result;
    int ret = glob("/opt/m5stack/bin/llm_*-*", GLOB_TILDE, NULL, &glob_result);  // 匹配所有.txt文件
    if (ret == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; i++) {
            const char *separator = strrchr(glob_result.gl_pathv[i], '/');
            const char *filename  = (separator != NULL) ? separator + 1 : glob_result.gl_pathv[i];
            data_body.push_back(std::string(filename));
        }
    }
    globfree(&glob_result);
    usr_out(json_obj["request_id"], json_obj["work_id"], data_body, com_id);
    int out = 0;
    return out;
}

int sys_reboot(int com_id, const nlohmann::json &json_obj)
{
    int out = 0;
    usr_print_error(json_obj["request_id"], json_obj["work_id"], "{\"code\":0, \"message\":\"rebooting ...\"}", com_id);
    usleep(200000);
    system("reboot");
    return out;
}

void server_work()
{
    key_sql["sys.ping"]      = sys_ping;
    key_sql["sys.lsmode"]    = sys_lsmode;
    key_sql["sys.lstask"]    = sys_lstask;
    key_sql["sys.push"]      = sys_push;
    key_sql["sys.pull"]      = sys_pull;
    key_sql["sys.update"]    = sys_update;
    key_sql["sys.upgrade"]   = sys_upgrade;
    key_sql["sys.bashexec"]  = sys_bashexec;
    key_sql["sys.hwinfo"]    = sys_hwinfo;
    key_sql["sys.uartsetup"] = sys_uartsetup;
    key_sql["sys.reset"]     = sys_reset;
    key_sql["sys.reboot"]    = sys_reboot;
    key_sql["sys.version"]   = sys_version;
    key_sql["sys.rmmode"]    = sys_rmmode;
    key_sql["sys.unit_call"] = sys_unit_call;
    key_sql["sys.cmminfo"]   = sys_cmminfo;
    key_sql["sys.version2"]  = sys_version2;
}

void server_stop_work()
{
}
std::mutex unit_action_match_mtx;
simdjson::ondemand::parser parser;
typedef int (*sys_fun_call)(int, const nlohmann::json &);
void unit_action_match(int com_id, const std::string &json_str)
{
    std::lock_guard<std::mutex> guard(unit_action_match_mtx);
    simdjson::padded_string json_string(json_str);
    simdjson::ondemand::document doc;
    auto error = parser.iterate(json_string).get(doc);
    if (error) {
        SLOGE("json format error:%s", json_str.c_str());
        usr_print_error("0", "sys", "{\"code\":-2, \"message\":\"json format error\"}", com_id);
        return;
    }
    std::string request_id;
    error = doc["request_id"].get_string(request_id);
    if (error) {
        SLOGE("miss request_id, error:%s", simdjson::error_message(error));
        usr_print_error("0", "sys", "{\"code\":-2, \"message\":\"json format error\"}", com_id);
        return;
    }
    std::string work_id;
    error = doc["work_id"].get_string(work_id);
    if (error) {
        SLOGE("miss work_id, error:%s", simdjson::error_message(error));
        usr_print_error("0", "sys", "{\"code\":-2, \"message\":\"json format error\"}", com_id);
        return;
    }
    if (work_id.empty()) work_id = "sys";
    std::string action;
    error = doc["action"].get_string(action);
    if (error) {
        SLOGE("miss action, error:%s", simdjson::error_message(error));
        usr_print_error("0", "sys", "{\"code\":-2, \"message\":\"json format error\"}", com_id);
        return;
    }
    // SLOGI("request_id:%s work_id:%s action:%s", request_id.c_str(), work_id.c_str(), action.c_str());
    std::vector<std::string> work_id_fragment;
    std::string fragment;
    for (auto c : work_id) {
        if (c != '.') {
            fragment.push_back(c);
        } else {
            work_id_fragment.push_back(fragment);
            fragment.clear();
        }
    }
    if (fragment.length()) work_id_fragment.push_back(fragment);
    if (action == "inference") {
        char zmq_push_url[128];
        int post = sprintf(zmq_push_url, zmq_c_format.c_str(), com_id);
        std::string inference_raw_data;
        inference_raw_data.resize(post + json_str.length() + 13);
        post = sprintf(inference_raw_data.data(), "{\"zmq_com\":\"");
        post += sprintf(inference_raw_data.data() + post, "%s", zmq_push_url);
        post += sprintf(inference_raw_data.data() + post, "\",");
        memcpy(inference_raw_data.data() + post, json_str.data() + 1, json_str.length() - 1);
        // post += json_str.length() - 1;
        // SLOGI("inference_raw_data:%s  size:%d  %d", inference_raw_data.c_str(), inference_raw_data.length(), post);
        int ret = zmq_bus_publisher_push(work_id, inference_raw_data);
        if (ret) {
            usr_print_error(request_id, work_id, "{\"code\":-4, \"message\":\"inference data push false\"}", com_id);
        }
    } else if ((work_id_fragment.size() > 0) && (work_id_fragment[0] == "sys")) {
        std::string unit_action = "sys." + action;
        sys_fun_call call_fun   = NULL;
        SAFE_READING(call_fun, sys_fun_call, unit_action);
        if (call_fun) {
            call_fun(com_id, nlohmann::json::parse(json_str));
        } else {
            usr_print_error(request_id, work_id, "{\"code\":-3, \"message\":\"action match false\"}", com_id);
        }
    } else {
        if ((work_id_fragment[0].length() != 0) && (remote_call(com_id, json_str) != 0)) {
            usr_print_error(request_id, work_id, "{\"code\":-9, \"message\":\"unit call false\"}", com_id);
        }
    }
}