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
#include <base64.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <modbus.h>
#include <arpa/inet.h>
// #include "cmdline.hpp"
#include <../../static_lib/include/fmt/core.h>
#include <../../static_lib/include/fmt/format.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_ec_prox will be exit!");
    main_exit_flage = 1;
}

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

using namespace StackFlows;

class llm_ec_prox : public StackFlow {
private:
    enum { EVENT_LOAD_CONFIG = EVENT_EXPORT + 1, EVENT_QUEUE_PLAY };
    std::string ec_prox_event_channel = "ipc:///tmp/llm/ec_prox.event.socket";
    modbus_t *modbus_ctx;
    std::mutex modbus_mtx_;
    typedef struct {
        int reg_index;    // 16位寄存器索引
        int byte_offset;  // 在寄存器中字节偏移（0-低字节，1-高字节）
    } RegPosition;
    std::unique_ptr<pzmq> pub_ctx_;
    std::string StackFlow_pack_temp =
        R"format({{ "created":{},"data":{},"object":{},"request_id":{},"work_id":"ax650_ec_proxy" }})format";
    std::string StackFlow_pack_all_temp =
        R"format({{ "created":{},"data":{},"error":{{"code":{},"message":{}}},"object":{},"request_id":{},"work_id":"ax650_ec_proxy" }})format";
    std::string return_success_result(std::string data, std::string object = "\"\"", std::string request_id = "\"\"")
    {
        return fmt::format(StackFlow_pack_temp, std::time(nullptr), data, object, request_id);
    }

    static llm_ec_prox *self;

    void loop()
    {
        uint8_t butt = 0;
        while (!main_exit_flage) {
            {
                std::unique_lock<std::mutex> lock(this->modbus_mtx_);
                uint8_t tab_rp_bits[2];
                int rc = modbus_read_input_bits(modbus_ctx, 4, 1, tab_rp_bits);
                if (rc == -1) {
                    fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
                } else {
                    if (tab_rp_bits[0] != butt) {
                        if (tab_rp_bits[0]) {
                            pub_ctx_->send_data(return_success_result("1"));
                        } else {
                            pub_ctx_->send_data(return_success_result("0"));
                        }
                        butt = tab_rp_bits[0];
                    }
                }
                // std::cout << "butt:" << (int)butt << std::endl;
            }
            usleep(20 * 1000);
        }
    }

    std::string board_get_power_info(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        // mbpoll -m rtu -b 115200 -P none -a 1 -r 1 -t 3 -c 13 -l 10 /dev/ttyS3
        uint16_t tab_reg[12];
        int rc = modbus_read_input_registers(modbus_ctx, 0, 12, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            return return_success_result(fmt::format(
                R"format({{ "pcie0_mv": {} , "pcie0_ma": {} , "pcie1_mv": {} , "pcie1_ma": {} , "usb1_mv": {} , "usb1_ma": {} , "usb2_mv": {} , "usb2_ma": {} , "VDD5V_mv": {} , "VDD5V_ma": {} , "EXT5V_mv": {} , "EXT5V_ma": {} }})format",
                tab_reg[0], tab_reg[1], tab_reg[2], tab_reg[3], tab_reg[4], tab_reg[5], tab_reg[6], tab_reg[7],
                tab_reg[8], tab_reg[9], tab_reg[10], tab_reg[11]));
        }
        return std::string("Error");
    }

    std::string fan_get_speed(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        // mbpoll -m rtu -b 115200 -P none -a 1 -r 1 -t 3 -c 13 -l 10 /dev/ttyS3
        uint16_t tab_reg[2];
        int rc = modbus_read_input_registers(modbus_ctx, 12, 1, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            return return_success_result(fmt::format("{}", tab_reg[0]));
        }
        return std::string("Error");
    }
    std::string fan_get_pwm(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        int rc = modbus_read_registers(modbus_ctx, 1, 1, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            return return_success_result(fmt::format("{}", tab_reg[0]));
        }
        return std::string("Error");
    }
    std::string fan_set_pwm(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_register(modbus_ctx, 1, tab_reg[0]);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
        }
        return return_success_result("\"ok\"");
    }

    RegPosition get_rgb_reg_pos(int rgb_index)
    {
        RegPosition res;
        int offset      = rgb_index * 3;  // 每个RGB占3字节
        res.reg_index   = offset / 2;
        res.byte_offset = offset % 2;
        return res;
    }
    std::string rgb_get_mode(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        int rc = modbus_read_registers(modbus_ctx, 11, 1, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            return return_success_result(fmt::format("{}", tab_reg[0]));
        }
        return std::string("Error");
    }
    std::string rgb_set_mode(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_register(modbus_ctx, 11, tab_reg[0]);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string rgb_get_size(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        int rc = modbus_read_registers(modbus_ctx, 10, 1, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            return return_success_result(fmt::format("{}", tab_reg[0]));
        }
        return std::string("Error");
    }
    std::string rgb_set_size(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_register(modbus_ctx, 10, tab_reg[0]);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string rgb_get_color(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        int rgb_index = 0;
        int rgb_color = 0;
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            rgb_index               = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        uint16_t tab_reg[2];
        RegPosition reg_pos = get_rgb_reg_pos(rgb_index);
        {
            int rc = modbus_read_registers(modbus_ctx, reg_pos.reg_index + 12, 2, tab_reg);
            if (rc == -1) {
                fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
            } else {
                for (int i = 0; i < 2; i++) tab_reg[i] = ntohs(tab_reg[i]);
                uint8_t *rgb_color_reg = (uint8_t *)tab_reg;
                rgb_color              = rgb_color_reg[reg_pos.byte_offset] << 16;
                rgb_color |= rgb_color_reg[reg_pos.byte_offset + 1] << 8;
                rgb_color |= rgb_color_reg[reg_pos.byte_offset + 2];
                return return_success_result(fmt::format("{}", rgb_color));
            }
        }
        return std::string("Error");
    }
    std::string rgb_set_color(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        int rgb_index = 0;
        int rgb_color = 0;
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            rgb_index               = (int)fan_data["data"]["rgb_index"];
            rgb_color               = (int)fan_data["data"]["rgb_color"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        uint16_t tab_reg[2];
        RegPosition reg_pos = get_rgb_reg_pos(rgb_index);
        {
            int rc = modbus_read_registers(modbus_ctx, reg_pos.reg_index + 12, 2, tab_reg);
            if (rc == -1) {
                fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
            }
            for (int i = 0; i < 2; i++) tab_reg[i] = ntohs(tab_reg[i]);
        }
        {
            uint8_t *rgb_color_reg                 = (uint8_t *)tab_reg;
            rgb_color_reg[reg_pos.byte_offset]     = (rgb_color & 0xff0000) >> 16;
            rgb_color_reg[reg_pos.byte_offset + 1] = (rgb_color & 0xff00) >> 8;
            rgb_color_reg[reg_pos.byte_offset + 2] = (rgb_color & 0xff);
        }
        for (int i = 0; i < 2; i++) tab_reg[i] = htons(tab_reg[i]);
        {
            int rc = modbus_write_registers(modbus_ctx, reg_pos.reg_index + 12, 2, tab_reg);
            if (rc == -1) {
                fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
                return std::string("Error");
            }
        }
        return return_success_result("\"ok\"");
    }

    std::string lcd_get_mode(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        int rc = modbus_read_registers(modbus_ctx, 86, 1, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            return return_success_result(fmt::format("{}", tab_reg[0]));
        }
        return std::string("Error");
    }
    std::string lcd_set_mode(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2];
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_register(modbus_ctx, 86, tab_reg[0]);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string lcd_set_ram(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        std::string rgb_ram_base64;
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            rgb_ram_base64          = fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        uint16_t *rgb_ram = (uint16_t *)malloc(((BASE64_DECODE_OUT_SIZE(rgb_ram_base64.length()) / 2) + 1) * 2);
        int ram_size      = base64_decode(rgb_ram_base64.c_str(), rgb_ram_base64.length(), (uint8_t *)rgb_ram);
        ram_size          = ram_size > 1870 ? 1870 : ram_size;
        int rc            = modbus_write_registers(modbus_ctx, 88, ram_size / 2, rgb_ram);
        free(rgb_ram);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return std::string("Ok");
    }

    std::string lcd_echo(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        std::string put_strs;
        uint16_t tab_reg = 0;
        uint8_t *c_reg   = (uint8_t *)&tab_reg;
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            put_strs                = fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        auto it = put_strs.begin();
        while (it != put_strs.end()) {
            c_reg[0] = *it;
            ++it;
            if (it != put_strs.end()) {
                c_reg[1] = *it;
                ++it;
            } else {
                c_reg[1] = 0;
            }
            int rc = modbus_write_register(modbus_ctx, 87, tab_reg);
            if (rc == -1) {
                fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
                return std::string("Error");
            }
        }
        return return_success_result("\"ok\"");
    }

    std::string ip_get(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint32_t value      = 0;
        uint16_t tab_reg[2] = {0};

        int rc = modbus_read_registers(modbus_ctx, 5, 2, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            for (int i = 0; i < 2; i++) tab_reg[i] = ntohs(tab_reg[i]);
            value = ((uint32_t)tab_reg[0] << 16) | tab_reg[1];
            char ip_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &value, ip_str, INET_ADDRSTRLEN);
            return return_success_result(fmt::format("\"{}\"", ip_str));
        }
        return std::string("Error");
    }
    std::string ip_set(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint32_t value;
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            std::string ip_str      = fan_data["data"];
            if (inet_pton(AF_INET, ip_str.c_str(), &value) != 1) {
                return std::string("Error : 无效的 IP 地址\n");
            }
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        uint16_t tab_reg[2];
        tab_reg[0] = (value >> 16) & 0xFFFF;
        tab_reg[1] = value & 0xFFFF;
        for (int i = 0; i < 2; i++) tab_reg[i] = htons(tab_reg[i]);
        int rc = modbus_write_registers(modbus_ctx, 5, 2, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

    std::string vddcpu_get(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};

        int rc = modbus_read_registers(modbus_ctx, 2, 1, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "读取失败: %s\n", modbus_strerror(errno));
        } else {
            return return_success_result(fmt::format("{}", tab_reg[0]));
        }
        return std::string("Error");
    }
    std::string vddcpu_set(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_registers(modbus_ctx, 2, 1, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string modbus_set_speed(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        int modbus_speed;
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            modbus_speed            = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        set_modbus_speed(modbus_speed);
        return return_success_result("\"ok\"");
    }
    std::string modbus_get_speed(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        return return_success_result(fmt::format("{}", _modbus_speed));
    }
    void set_modbus_speed(int modbus_speed)
    {
        uint16_t tab_reg[2];
        tab_reg[0] = (modbus_speed >> 16) & 0xFFFF;
        tab_reg[1] = modbus_speed & 0xFFFF;
        int rc     = modbus_write_registers(modbus_ctx, 3, 2, tab_reg);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            std::exit(-1);
        }
        modbus_close(modbus_ctx);
        modbus_free(modbus_ctx);
        modbus_ctx = NULL;
        usleep(100 * 1000);
        modbus_ctx = modbus_new_rtu("/dev/ttyS3", modbus_speed, 'N', 8, 1);
        if (modbus_ctx == NULL) {
            fprintf(stderr, "Unable to create the context\n");
            std::exit(-1);
        }
        modbus_set_slave(modbus_ctx, 1);  // 设置 Modbus 从站地址为1
        if (modbus_connect(modbus_ctx) == -1) {
            fprintf(stderr, "Connection failed: %s\n", modbus_strerror(errno));
            modbus_free(modbus_ctx);
            std::exit(-1);
        }
        _modbus_speed = modbus_speed;
    }
    int _modbus_speed = 115200;

    void _init_modbus()
    {
        modbus_ctx = modbus_new_rtu("/dev/ttyS3", 115200, 'N', 8, 1);
        if (modbus_ctx == NULL) {
            fprintf(stderr, "Unable to create the context\n");
            std::exit(-1);
        }
        modbus_set_slave(modbus_ctx, 1);  // 设置 Modbus 从站地址为1
        if (modbus_connect(modbus_ctx) == -1) {
            fprintf(stderr, "Connection failed: %s\n", modbus_strerror(errno));
            modbus_free(modbus_ctx);
            std::exit(-1);
        }
        uint8_t tab_rp_bits[2];
        int rc = modbus_read_input_bits(modbus_ctx, 4, 1, tab_rp_bits);
        if (rc == -1) {
            _modbus_speed = 1152000;
            modbus_ctx    = modbus_new_rtu("/dev/ttyS3", _modbus_speed, 'N', 8, 1);
            if (modbus_ctx == NULL) {
                fprintf(stderr, "Unable to create the context\n");
                std::exit(-1);
            }
            modbus_set_slave(modbus_ctx, 1);  // 设置 Modbus 从站地址为1
            if (modbus_connect(modbus_ctx) == -1) {
                fprintf(stderr, "Connection failed: %s\n", modbus_strerror(errno));
                modbus_free(modbus_ctx);
                std::exit(-1);
            }
            uint8_t tab_rp_bits[2];
            int rc = modbus_read_input_bits(modbus_ctx, 4, 1, tab_rp_bits);
            if (rc == -1) {
                fprintf(stderr, "modbus 115200/1152000 速度探测失败: %s\n", modbus_strerror(errno));
                std::exit(-1);
            }
        } else {
            set_modbus_speed(1152000);
        }
    }

    std::string ext_poweroff(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 0, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string board_poweroff(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 1, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

    std::string pcie0_set_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 4, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string pcie1_set_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 5, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

    std::string gl3510_reset(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 6, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string usbds1_set_big_power(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 7, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string usbds2_set_big_power(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 8, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string usbds1_set_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 9, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string usbds2_set_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 10, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string usbds3_set_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 11, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

    std::string grove_uart_set_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 12, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }
    std::string grove_iic_set_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 13, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

    std::string flash_save_switch(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 14, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

    std::string flash_save_value_config(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_bit(modbus_ctx, 15, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

    std::string poweroff(StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data)
    {
        uint16_t tab_reg[2] = {0};
        try {
            nlohmann::json fan_data = nlohmann::json::parse(data->string());
            tab_reg[0]              = (int)fan_data["data"];
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "解析失败：" << e.what() << std::endl;
            return std::string("Error");
        } catch (...) {
            std::cout << "data 解析异常" << std::endl;
            return std::string("Error");
        }
        int rc = modbus_write_register(modbus_ctx, 0, tab_reg[0] >= 1000 ? tab_reg[0] : 1000);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        rc = modbus_write_bit(modbus_ctx, 16, 1);
        if (rc == -1) {
            fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            return std::string("Error");
        }
        return return_success_result("\"ok\"");
    }

public:
    llm_ec_prox() : StackFlow("ec_prox")
    {
        // setup("", "audio.play", "{\"None\":\"None\"}");
        self = this;
        // prob modbus speed
        _init_modbus();

        pub_ctx_ = std::make_unique<pzmq>(ec_prox_event_channel, ZMQ_PUB);

        // clang-format off
#define REGISTER_RPC_ACTION(name, func) \
        rpc_ctx_->register_rpc_action(name, [this](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data){ \
            std::unique_lock<std::mutex> lock(this->modbus_mtx_); \
            return this->func(_pzmq, data); \
        });

        REGISTER_RPC_ACTION("fan_get_speed",                fan_get_speed);
        REGISTER_RPC_ACTION("fan_get_pwm",                  fan_get_pwm);
        REGISTER_RPC_ACTION("fan_set_pwm",                  fan_set_pwm);

        REGISTER_RPC_ACTION("rgb_get_mode",                 rgb_get_mode);
        REGISTER_RPC_ACTION("rgb_set_mode",                 rgb_set_mode);
        REGISTER_RPC_ACTION("rgb_get_size",                 rgb_get_size);
        REGISTER_RPC_ACTION("rgb_set_size",                 rgb_set_size);
        REGISTER_RPC_ACTION("rgb_get_color",                rgb_get_color);
        REGISTER_RPC_ACTION("rgb_set_color",                rgb_set_color);
        
        REGISTER_RPC_ACTION("board_get_power_info",         board_get_power_info);

        REGISTER_RPC_ACTION("ip_get",                       ip_get);
        REGISTER_RPC_ACTION("ip_set",                       ip_set);

        REGISTER_RPC_ACTION("lcd_get_mode",                 lcd_get_mode);
        REGISTER_RPC_ACTION("lcd_set_mode",                 lcd_set_mode);
        REGISTER_RPC_ACTION("lcd_set_ram",                  lcd_set_ram);
        REGISTER_RPC_ACTION("lcd_echo",                     lcd_echo);
        

        REGISTER_RPC_ACTION("vddcpu_get",                   vddcpu_get);
        REGISTER_RPC_ACTION("vddcpu_set",                   vddcpu_set);

        REGISTER_RPC_ACTION("modbus_get_speed",             modbus_get_speed);
        REGISTER_RPC_ACTION("modbus_set_speed",             modbus_set_speed);

        REGISTER_RPC_ACTION("ext_poweroff",                   ext_poweroff);
        REGISTER_RPC_ACTION("board_poweroff",                 board_poweroff);
        REGISTER_RPC_ACTION("pcie0_set_switch",                 pcie0_set_switch);
        REGISTER_RPC_ACTION("pcie1_set_switch",                 pcie1_set_switch);
        REGISTER_RPC_ACTION("gl3510_reset",                 gl3510_reset);
        REGISTER_RPC_ACTION("usbds1_set_big_power",                 usbds1_set_big_power);
        REGISTER_RPC_ACTION("usbds2_set_big_power",                 usbds2_set_big_power);
        REGISTER_RPC_ACTION("usbds1_set_switch",                 usbds1_set_switch);
        REGISTER_RPC_ACTION("usbds2_set_switch",                 usbds2_set_switch);
        REGISTER_RPC_ACTION("usbds3_set_switch",                 usbds3_set_switch);
        REGISTER_RPC_ACTION("grove_uart_set_switch",                 grove_uart_set_switch);
        REGISTER_RPC_ACTION("grove_iic_set_switch",                 grove_iic_set_switch);
        REGISTER_RPC_ACTION("flash_save_switch",                 flash_save_switch);
        REGISTER_RPC_ACTION("flash_save_value_config",                 flash_save_value_config);
        REGISTER_RPC_ACTION("poweroff",                 poweroff);




#undef REGISTER_RPC_ACTION
        // clang-format on
    }
    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        send(std::string("None"), std::string("None"), std::string(""), unit_name_);
        return -1;
    }
    void ax650_ec_prox_exit()
    {
        pub_ctx_.reset();
        for (int i = 0; i < 3; i++) {
            uint32_t value = 115200;
            uint16_t tab_reg[2];
            tab_reg[0] = (value >> 16) & 0xFFFF;
            tab_reg[1] = value & 0xFFFF;
            int rc     = modbus_write_registers(modbus_ctx, 3, 2, tab_reg);
            if (rc == -1) {
                fprintf(stderr, "写入失败: %s\n", modbus_strerror(errno));
            } else {
                break;
            }
        }
        modbus_close(modbus_ctx);
        modbus_free(modbus_ctx);
        std::cout << "all exit!" << std::endl;
    }
    ~llm_ec_prox()
    {
        ax650_ec_prox_exit();
    }
};

llm_ec_prox *llm_ec_prox::self;
int main(int argc, char *argv[])
{
    signal(SIGTERM, __sigint);
    signal(SIGINT, __sigint);
    mkdir("/tmp/llm", 0777);
    llm_ec_prox ec_prox;
    while (!main_exit_flage) {
        sleep(1);
    }
    ec_prox.llm_firework_exit();
    return 0;
}
