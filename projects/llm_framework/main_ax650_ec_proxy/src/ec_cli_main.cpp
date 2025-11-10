#include <iostream>
#include <modbus.h>
#include "pzmq.hpp"
#include "cmdline.hpp"
#include <../../static_lib/include/fmt/core.h>
#include <../../static_lib/include/fmt/format.h>
#include <cstdlib>
#include <list>
#include <vector>
#include "json.hpp"
int condition = 1;
const char *rpc_socket_path;
const char *pub_socket_path;

void info_fun(cmdline::parser &a, std::string set_fun, std::string get_fun)
{
    if (a.exist("data") || a.exist("DataRaw")) {
        if (set_fun.length() == 0) return;
        std::string DataRaw = a.get<std::string>("DataRaw");
        if (DataRaw.empty()) {
            DataRaw = fmt::format("{{ \"data\": {} }}", a.get<std::string>("data"));
        }
        StackFlows::pzmq Context(rpc_socket_path);
        Context.call_rpc_action(set_fun, DataRaw,
                                [set_fun](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        std::cout << data->string() << std::endl;
                                    } else {
                                        std::cout << "call " << set_fun << " faile!" << std::endl;
                                    }
                                });
    } else {
        if (get_fun.length() == 0) {
            std::cout << "please set data: -d or -D" << std::endl;
            return;
        }
        StackFlows::pzmq Context(rpc_socket_path);
        Context.call_rpc_action(get_fun, "{}",
                                [get_fun](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        std::cout << data->string() << std::endl;
                                    } else {
                                        std::cout << "call " << get_fun << " faile! " << std::endl;
                                    }
                                });
    }
}

void get_info_fun(cmdline::parser &a, std::string fun)
{
    StackFlows::pzmq Context(rpc_socket_path);
    Context.call_rpc_action(fun, "{\"data\":\"None\"}",
                            [fun](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                if (data->size() > 0) {
                                    std::cout << data->string() << std::endl;
                                } else {
                                    std::cout << "call " << fun << " faile!" << std::endl;
                                }
                            });
}

void set_info_fun(cmdline::parser &a, std::string fun)
{
    if (a.exist("data") || a.exist("DataRaw")) {
        std::string DataRaw = a.get<std::string>("DataRaw");
        if (DataRaw.empty()) {
            DataRaw = fmt::format("{{ \"data\": {} }}", a.get<std::string>("data"));
        }
        StackFlows::pzmq Context(rpc_socket_path);
        Context.call_rpc_action(fun, DataRaw,
                                [fun](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        std::cout << data->string() << std::endl;
                                    } else {
                                        std::cout << "call " << fun << "faile!" << std::endl;
                                    }
                                });
    }
}

void fan_speed_fun(cmdline::parser &a)
{
    StackFlows::pzmq Context(rpc_socket_path);
    Context.call_rpc_action("fan_get_speed", "{}",
                            [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                if (data->size() > 0) {
                                    std::cout << data->string() << std::endl;
                                } else {
                                    std::cout << "call fan_get_speed faile!" << std::endl;
                                }
                            });
}

void fan_pwm_fun(cmdline::parser &a)
{
    if (a.exist("data") || a.exist("DataRaw")) {
        std::string DataRaw = a.get<std::string>("DataRaw");
        if (DataRaw.empty()) {
            DataRaw = fmt::format("{{ \"data\": {} }}", a.get<std::string>("data"));
        }
        StackFlows::pzmq Context(rpc_socket_path);
        Context.call_rpc_action("fan_set_pwm", DataRaw,
                                [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        std::cout << data->string() << std::endl;
                                    } else {
                                        std::cout << "call fan_set_pwm faile!" << std::endl;
                                    }
                                });
    } else {
        StackFlows::pzmq Context(rpc_socket_path);
        Context.call_rpc_action("fan_get_pwm", "{}",
                                [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        std::cout << data->string() << std::endl;
                                    } else {
                                        std::cout << "call fan_get_pwm faile!" << std::endl;
                                    }
                                });
    }
}

void rgb_fun(cmdline::parser &a)
{
    if (a.exist("data") || a.exist("DataRaw")) {
        std::string DataRaw = a.get<std::string>("DataRaw");
        if (DataRaw.empty()) {
            DataRaw = fmt::format("{{ \"data\": {} }}", a.get<std::string>("data"));
        }
        StackFlows::pzmq Context(rpc_socket_path);
        Context.call_rpc_action("rgb_set_mode", DataRaw,
                                [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        std::cout << data->string() << std::endl;
                                    } else {
                                        std::cout << "call rgb_set_mode faile!" << std::endl;
                                    }
                                });
    } else {
        StackFlows::pzmq Context(rpc_socket_path);
        Context.call_rpc_action("rgb_get_mode", "{}",
                                [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        std::cout << data->string() << std::endl;
                                    } else {
                                        std::cout << "call rgb_get_mode faile!" << std::endl;
                                    }
                                });
    }
}

void exec_fun(cmdline::parser &a)
{
    if (a.exist("list")) {
        StackFlows::pzmq Context(rpc_socket_path);
        std::cout << "list" << std::endl;
        Context.call_rpc_action("list_action", "None",
                                [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                    if (data->size() > 0) {
                                        try {
                                            nlohmann::json _data = nlohmann::json::parse(data->string());
                                            std::cout << _data.dump(4) << std::endl;
                                        } catch (const nlohmann::json::parse_error &e) {
                                            std::cerr << "解析失败：" << e.what() << std::endl;
                                        }
                                    } else {
                                        std::cout << "call list_action faile!" << std::endl;
                                    }
                                });
        return;
    }

    const std::string fun  = a.get<std::string>("fun");
    const std::string data = a.get<std::string>("data");
    if (fun.length() == 0) {
        std::cout << "fun is empty! cli exec -f <fun> -d <data>" << std::endl;
        return;
    }

    StackFlows::pzmq Context(rpc_socket_path);
    Context.call_rpc_action(fun, data, [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
        if (data->size() > 0) {
            std::cout << "" << data->string() << std::endl;
        } else {
            std::cout << "call ax650_ec_prox faile!" << std::endl;
        }
    });
}

void echo_fun(cmdline::parser &a)
{
    if (a.exist("button")) {
        StackFlows::pzmq Context(pub_socket_path, ZMQ_SUB,
                                 [](StackFlows::pzmq *_pzmq, const std::shared_ptr<StackFlows::pzmq_data> &data) {
                                     std::cout << data->string() << std::endl;
                                 });
        while (condition) {
            usleep(100 * 1000);
        }
        exit(0);
    }
}

void signalHandler(int signum)
{
    condition = 0;
    std::exit(0);  // 这会调用全局对象的析构函数
}

struct call_fun {
    std::string fun;
    char flage;
    std::string dec;
    std::function<void(cmdline::parser &a)> fun_call;
};

int main(int argc, char *argv[])
{
    if (argc < 2) {
    error_print_exit:
        std::cout << R"help(
用法：git [--version] [--help] 
           <命令> [<参数>]

这些是各种场合常见的 cli 命令：

直接操作 ec （参见：cli help device）
   device    直接执行一个操作，使用指令传递参数

自定义操作 ec （参见：git help exec）
   exec       执行一个原始函数调用

订阅 prox 的通道数据（参见：git help echo）
   echo      显示 prox 的通道数据

命令 'cli help -a' 和 'cli help -g' 显示可用的子命令和一些概念帮助。
查看 'cli help <命令>' 或 'cli help <概念>' 以获取给定子命令或概念的
帮助。
)help" << std::endl;
        return 0;
    }
    signal(SIGTERM, signalHandler);
    signal(SIGINT, signalHandler);
    rpc_socket_path = getenv("AX650C_EC_PROXY_RPC_SOCKET");
    if (rpc_socket_path == NULL) {
        rpc_socket_path = "ipc:///tmp/rpc.ec_prox";
    }
    pub_socket_path = getenv("AX650C_EC_PROXY_PUB_SOCKET");
    if (pub_socket_path == NULL) {
        pub_socket_path = "ipc:///tmp/llm/ec_prox.event.socket";
    }
    if (std::string(argv[1]) == "device") {
        cmdline::parser a;
        // clang-format off
        std::list<struct call_fun> cmd_list = {
            {"rgb", 'r', "rgb_mode", std::bind(&info_fun, std::placeholders::_1, "rgb_set_mode", "rgb_get_mode")},
            {"rgb_size", 0, "rgb_size", std::bind(&info_fun, std::placeholders::_1, "rgb_set_size", "rgb_get_size")},
            {"rgb_get_color", 0, "rgb_get_color", std::bind(&info_fun, std::placeholders::_1, "rgb_get_color", "")},
            {"rgb_set_color", 0, "rgb set color ,example: cli --rgb_set_color -d '{\"rgb_index\":0,\"rgb_color\":255}'", std::bind(&info_fun, std::placeholders::_1, "rgb_set_color", "")},
            {"fan", 'f', "fan_pwm", std::bind(&info_fun, std::placeholders::_1, "fan_set_pwm", "fan_get_pwm")},
            {"fanspeed", 'F', "fan_speed", std::bind(&info_fun, std::placeholders::_1, "", "fan_get_speed")},
            {"board", 'B', "board_get_power_info", std::bind(&info_fun, std::placeholders::_1, "", "board_get_power_info")},
            {"ip", 'i', "ip fun", std::bind(&info_fun, std::placeholders::_1, "ip_set", "ip_get")},
            {"lcd", 'l', "lcd fun", std::bind(&info_fun, std::placeholders::_1, "lcd_set_mode", "lcd_get_mode")},
            {"lcd_ram", 0, "lcd_ram", std::bind(&info_fun, std::placeholders::_1, "lcd_set_ram", "")},
            {"lcd_echo", 'E', "lcd_echo", std::bind(&info_fun, std::placeholders::_1, "lcd_echo", "")},
            {"vddcpu", 'c', "vddcpu fun", std::bind(&info_fun, std::placeholders::_1, "vddcpu_set", "vddcpu_get")},
            {"modbus_speed", 0, "modbus_speed", std::bind(&info_fun, std::placeholders::_1, "modbus_set_speed", "modbus_get_speed")},
            {"ext_power", 'p', "ext_power fun", std::bind(&info_fun, std::placeholders::_1, "ext_power", "")},
            {"board_power", 'b', "board_power fun", std::bind(&info_fun, std::placeholders::_1, "board_power", "")},
            {"pcie0", 0, "pcie0_set_switch fun", std::bind(&info_fun, std::placeholders::_1, "pcie0_set_switch", "")},
            {"pcie1", 0, "pcie1_set_switch fun", std::bind(&info_fun, std::placeholders::_1, "pcie1_set_switch", "")},
            {"gl3510_reset", 0, "gl3510_reset fun", std::bind(&info_fun, std::placeholders::_1, "gl3510_reset", "")},
            {"usbds1_big", 0, "usbds1_set_big_power fun", std::bind(&info_fun, std::placeholders::_1, "usbds1_set_big_power", "")},
            {"usbds2_big", 0, "usbds2_set_big_power fun", std::bind(&info_fun, std::placeholders::_1, "usbds2_set_big_power", "")},
            {"usbds1", 0, "usbds1_set_switch fun", std::bind(&info_fun, std::placeholders::_1, "usbds1_set_switch", "")},
            {"usbds2", 0, "usbds2_set_switch fun", std::bind(&info_fun, std::placeholders::_1, "usbds2_set_switch", "")},
            {"usbds3", 0, "usbds3_set_switch fun", std::bind(&info_fun, std::placeholders::_1, "usbds3_set_switch", "")},
            {"grove_uart", 0, "grove_uart_set_switch fun", std::bind(&info_fun, std::placeholders::_1, "grove_uart_set_switch", "")},
            {"grove_iic", 0, "grove_iic_set_switch fun", std::bind(&info_fun, std::placeholders::_1, "grove_iic_set_switch", "")},
            {"flash_switch", 0, "flash_save_switch fun", std::bind(&info_fun, std::placeholders::_1, "flash_save_switch", "")},
            {"flash_value", 0, "flash_save_value_config fun", std::bind(&info_fun, std::placeholders::_1, "flash_save_value_config", "")},
            {"poweroff", 0, "poweroff fun", std::bind(&info_fun, std::placeholders::_1, "poweroff", "")},

        };
        // clang-format on
        for (auto &cmd : cmd_list) {
            a.add(cmd.fun, cmd.flage, cmd.dec);
        }
        a.add<std::string>("data", 'd', "call param", false);
        a.add<std::string>("DataRaw", 'D', "call param raw", false);
        a.parse_check(argc, argv);
        for (auto &cmd : cmd_list) {
            if (a.exist(cmd.fun)) {
                cmd.fun_call(a);
            }
        }
    } else if (std::string(argv[1]) == "exec") {
        cmdline::parser a;
        a.add("list", 'l', "list call function");
        a.add<std::string>("fun", 'f', "call ax650c_ec_proxy function", false);
        a.add<std::string>("data", 'd', "call ax650c_ec_proxy function input data", false);
        a.parse_check(argc, argv);
        exec_fun(a);
    } else if (std::string(argv[1]) == "echo") {
        cmdline::parser a;
        a.add("button", 'b', "button event");
        a.parse_check(argc, argv);
        echo_fun(a);
        std::cout << a.usage() << std::endl;
        return 0;
    } else {
        goto error_print_exit;
    }

    return 0;
}
