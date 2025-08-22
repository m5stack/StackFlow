#include "Tokenizer.hpp"

#include "sentencepiece_processor.h"
#include "builtin_pb/sentencepiece.pb.h"

#include "QwenTokenizer.hpp"

// #include "chatglm.h"

#include "httplib.h"
#include "http_utils.hpp"
#include "json.hpp"

#include "sample_log.h"
#include "string_utility.hpp"
#include "memory_utils.hpp"

#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <cstring>
#include <csignal>

class TokenizerLLaMa : public BaseTokenizer {
protected:
    sentencepiece::SentencePieceProcessor sp;
    bool _b_bos, _b_eos;

private:
    /* data */
public:
    bool Init(std::string model_path, bool b_bos = true, bool b_eos = false) override
    {
        auto ret = sp.Load(model_path);
        if (!ret.ok()) {
            ALOGE("%s", ret.error_message());
            return false;
        }

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return ret.ok();
    }

    bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) override
    {
        auto ret = sp.Encode(input, &output);
        if (!ret.ok()) {
            ALOGE("%s", ret.error_message());
            return false;
        }
        if (_b_bos) {
            output.insert(output.begin(), sp.bos_id());
        }
        if (_b_eos) {
            output.push_back(sp.eos_id());
        }
        return true;
    }

    std::vector<int> Encode(std::string input, ImageInfo img_info) override
    {
        std::vector<int> output;
        Encode(input, output, img_info);
        return output;
    }

    std::string Decode(const std::vector<int> &input) override
    {
        sentencepiece::SentencePieceText spt;
        sp.Decode(input, &spt);
        std::string out = spt.pieces()[0].piece();
        if (*(unsigned short *)out.data() == 38626) {
            return " " + spt.text();
        } else {
            return spt.text();
        }
    }

    int GetBosID() override
    {
        return sp.bos_id();
    }

    int GetEosID() override
    {
        return sp.eos_id();
    }
    std::string apply_chat_template() override
    {
        std::ostringstream oss_prompt;
        int messages_len = messages_.size();
        for (auto &message : messages_) {
            messages_len--;
            switch (message.first) {
                case ROLE_USER: {
                    oss_prompt << "<|user|>\n" << message.second << "</s>";
                } break;
                case ROLE_SYSTEM:
                    break;
                case ROLE_ASSISTANT:
                    break;
                case ROLE_ASSISTANT_HELP: {
                    if (messages_len == 0) {
                        oss_prompt << "<|assistant|>\n";
                    }
                } break;
                default:
                    break;
            }
        }
        return oss_prompt.str();
    }
};

class TokenizerMINICPM : public TokenizerLLaMa {
public:
    std::string Decode(const std::vector<int> &input) override
    {
        sentencepiece::SentencePieceText spt;
        sp.Decode(input, &spt);
        return spt.text();
    }
    std::string apply_chat_template() override
    {
        std::ostringstream oss_prompt;
        int messages_len = messages_.size();
        for (auto &message : messages_) {
            messages_len--;
            switch (message.first) {
                case ROLE_USER: {
                    oss_prompt << "<用户>" << message.second;
                } break;
                case ROLE_SYSTEM:
                    break;
                case ROLE_ASSISTANT:
                    break;
                case ROLE_ASSISTANT_HELP: {
                    if (messages_len == 0) {
                        oss_prompt << "<AI>";
                    }
                } break;
                default:
                    break;
            }
        }
        return oss_prompt.str();
    }
};

class TokenizerPhi3 : public BaseTokenizer {
    sentencepiece::SentencePieceProcessor sp;
    bool _b_bos, _b_eos;

private:
    /* data */
public:
    bool Init(std::string model_path, bool b_bos = true, bool b_eos = false) override
    {
        auto ret = sp.Load(model_path);
        if (!ret.ok()) {
            ALOGE("%s", ret.error_message());
            return false;
        }

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return ret.ok();
    }

    bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) override
    {
        auto ret = sp.Encode(input, &output);
        if (!ret.ok()) {
            ALOGE("%s", ret.error_message());
            return false;
        }
        output.insert(output.begin(), 32010);  //"<|user|>"
        output.push_back(32007);               //"<|end|>"
        output.push_back(32001);               //"<|assistant|>"
        if (_b_bos) {
            output.insert(output.begin(), sp.bos_id());
        }
        if (_b_eos) {
            output.push_back(sp.eos_id());
        }
        return true;
    }

    std::vector<int> Encode(std::string input, ImageInfo img_info) override
    {
        std::vector<int> output;
        Encode(input, output, img_info);
        return output;
    }

    std::string Decode(const std::vector<int> &input) override
    {
        sentencepiece::SentencePieceText spt;
        sp.Decode(input, &spt);
        std::string out = spt.pieces()[0].piece();
        if (*(unsigned short *)out.data() == 38626) {
            return " " + spt.text();
        } else {
            return spt.text();
        }
    }

    int GetBosID() override
    {
        return sp.bos_id();
    }

    int GetEosID() override
    {
        return 32007;
    }

    bool isEnd(int id) override
    {
        return id == GetEosID() || id > 31999;
    }

    std::string apply_chat_template() override
    {
        std::ostringstream oss_prompt;
        int messages_len = messages_.size();
        for (auto &message : messages_) {
            messages_len--;
            switch (message.first) {
                case ROLE_USER: {
                    oss_prompt << message.second;
                } break;
                case ROLE_SYSTEM:
                    break;
                case ROLE_ASSISTANT:
                    break;
                case ROLE_ASSISTANT_HELP: {
                    if (messages_len == 0) {
                        oss_prompt << " ";
                    }
                } break;
                default:
                    break;
            }
        }
        return oss_prompt.str();
    }
};

class TokenizerQwen : public BaseTokenizer {
    std::shared_ptr<QwenTokenizer> sp;
    bool _b_bos, _b_eos;

private:
    /* data */
public:
    bool Init(std::string model_path, bool b_bos = true, bool b_eos = false) override
    {
        if (!file_exist(model_path)) {
            ALOGE("tokenizer model file(%s) not exist", model_path.c_str());
            return false;
        }

        sp.reset(new QwenTokenizer(model_path, QwenConfig()));

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return true;
    }

    bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) override
    {
        if (_b_bos) {
            // input += "<|im_start|>";
        }
        if (_b_eos) {
            input += "<|endoftext|>";
        }
        output = sp->encode(input, 1024);

        return true;
    }

    std::vector<int> Encode(std::string input, ImageInfo img_info) override
    {
        std::vector<int> output;
        Encode(input, output, img_info);
        return output;
    }

    std::string Decode(const std::vector<int> &input) override
    {
        return sp->decode(input);
    }

    int GetBosID() override
    {
        return -1;
    }

    int GetEosID() override
    {
        return sp->eos_token_id;
    }
    std::string apply_chat_template() override
    {
        std::ostringstream oss_prompt;
        int messages_len = messages_.size();
        for (auto &message : messages_) {
            messages_len--;
            switch (message.first) {
                case ROLE_USER: {
                    oss_prompt << "<|im_start|>user\n" << message.second << "<|im_end|>\n";
                } break;
                case ROLE_SYSTEM: {
                    oss_prompt << "<|im_start|>system\n" << message.second << ".<|im_end|>\n";
                } break;
                case ROLE_ASSISTANT: {
                    oss_prompt << "<|im_start|>assistant\n" << message.second << ".<|im_end|>\n";
                } break;
                case ROLE_ASSISTANT_HELP: {
                    if (messages_len == 0) {
                        oss_prompt << "<|im_start|>assistant\n";
                    }
                } break;
                default:
                    break;
            }
        }
        return oss_prompt.str();
    }
};

class Tokenizer_Http : public BaseTokenizer {
    std::shared_ptr<httplib::Client> cli;
    bool _b_bos, _b_eos;

    std::string base_url;

    int bos_id, eos_id;

    std::string uid;

private:
    /* data */
public:
    bool Init(std::string model_path = "http://localhost:8080", bool b_bos = true, bool b_eos = false) override
    {
        base_url = model_path;
        try {
            cli = std::make_shared<httplib::Client>(base_url);
            cli->set_connection_timeout(1);
            cli->set_read_timeout(1);
            cli->set_write_timeout(1);
            {
                auto ret = cli->Get("/bos_id");
                auto rep = ret.value();
                if (rep.status != 200) {
                    ALOGE("get bos_id failed, status: %d", rep.status);
                    return false;
                }
                nlohmann::json j = nlohmann::json::parse(rep.body);
                bos_id           = j["bos_id"];
            }

            {
                auto ret = cli->Get("/eos_id");
                auto rep = ret.value();
                if (rep.status != 200) {
                    ALOGE("get eos_id failed, status: %d", rep.status);
                    return false;
                }
                nlohmann::json j = nlohmann::json::parse(rep.body);
                eos_id           = j["eos_id"];
            }
            printf("bos_id: %d, eos_id: %d\n", bos_id, eos_id);
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            return false;
        }

        this->_b_bos = b_bos;
        this->_b_eos = b_eos;
        return true;
    }

    bool Init(std::string model_path = "http://localhost:8080") override
    {
        base_url = model_path;
        if (!test_connect_http(base_url, 10)) {
            ALOGE("connect %s failed", base_url.c_str());
            return false;
        } else {
            ALOGI("connect %s ok", base_url.c_str());
        }

        cli = std::make_shared<httplib::Client>(base_url);
        cli->set_connection_timeout(10);
        cli->set_read_timeout(10);
        cli->set_write_timeout(10);

        int try_count = 10;
        int count     = try_count;
        while (count-- > 0) {
            try {
                auto ret = cli->Get("/get_uid");
                auto rep = ret.value();
                if (rep.status != 200) {
                    ALOGE("get uid failed, status: %d", rep.status);
                    return false;
                }
                nlohmann::json j = nlohmann::json::parse(rep.body);
                uid              = j["uid"];
                ALOGI("uid: %s", uid.c_str());
                break;
            } catch (const std::exception &e) {
                std::cerr << e.what() << '\n';
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            ALOGE("get uid failed, try again %d/%d", count, try_count);
        }

        count = 10;
        while (count-- > 0) {
            try {
                auto ret = cli->Get("/bos_id?uid=" + uid);
                auto rep = ret.value();
                if (rep.status != 200) {
                    ALOGE("get bos_id failed, status: %d", rep.status);
                    return false;
                }
                nlohmann::json j = nlohmann::json::parse(rep.body);
                bos_id           = j["bos_id"];
                break;
            } catch (const std::exception &e) {
                std::cerr << e.what() << '\n';
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            ALOGE("get bos_id failed, try again %d/%d", count, try_count);
        }

        count = 10;
        while (count-- > 0) {
            try {
                auto ret = cli->Get("/eos_id?uid=" + uid);
                auto rep = ret.value();
                if (rep.status != 200) {
                    ALOGE("get eos_id failed, status: %d", rep.status);
                    return false;
                }
                nlohmann::json j = nlohmann::json::parse(rep.body);
                eos_id           = j["eos_id"];
                break;
            } catch (const std::exception &e) {
                std::cerr << e.what() << '\n';
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            ALOGE("get eos_id failed, try again %d/%d", count, try_count);
        }

        printf("bos_id: %d, eos_id: %d\n", bos_id, eos_id);

        return true;
    }

    bool Encode(std::string input, std::string last_reply, std::vector<int> &tokens, std::vector<int> &tokens_diff,
                ImageInfo img_info) override
    {
        nlohmann::json j;
        j["uid"]  = uid;
        j["text"] = input;
        if (!last_reply.empty() and last_reply != "") {
            j["last_reply"] = last_reply;
        }

        j["img_prompt"] = img_info.img_prompt;
        auto ret        = cli->Post("/encode", j.dump(), "application/json");
        auto rep        = ret.value();
        if (rep.status != 200) {
            ALOGE("encode failed, status: %d", rep.status);
            return false;
        }
        nlohmann::json j2;
        try {
            j2 = nlohmann::json::parse(rep.body);
        } catch (const std::exception &e) {
            ALOGE("json parse failed: %s", e.what());
            ALOGE("%s", rep.body.c_str());
            return false;
        }

        std::vector<int> _token_ids   = j2["token_ids"];
        std::vector<int> _tokens_diff = j2["diff"];

        tokens      = _token_ids;
        tokens_diff = _tokens_diff;

        return true;
    }

    bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) override
    {
        nlohmann::json j;
        j["text"]       = input;
        j["img_prompt"] = img_info.img_prompt;
        auto ret        = cli->Post("/encode", j.dump(), "application/json");
        auto rep        = ret.value();
        if (rep.status != 200) {
            ALOGE("encode failed, status: %d", rep.status);
            return false;
        }
        nlohmann::json j2;
        try {
            j2 = nlohmann::json::parse(rep.body);
        } catch (const std::exception &e) {
            ALOGE("json parse failed: %s", e.what());
            ALOGE("%s", rep.body.c_str());
            return false;
        }

        std::vector<int> out = j2["token_ids"];
        output               = out;

        if (_b_bos) {
            output.insert(output.begin(), bos_id);
        }
        if (_b_eos) {
            output.push_back(eos_id);
        }

        return true;
    }

    bool Reset(std::string system_prompt, std::vector<int> &tokens) override
    {
        nlohmann::json j;
        j["uid"] = uid;
        if (!system_prompt.empty() and system_prompt != "") {
            j["system_prompt"] = system_prompt;
        }

        auto ret = cli->Post("/reset", j.dump(), "application/json");
        auto rep = ret.value();
        if (rep.status != 200) {
            ALOGE("reset failed, status: %d", rep.status);
            return false;
        }
        nlohmann::json j_rep        = nlohmann::json::parse(rep.body);
        std::vector<int> _token_ids = j_rep["token_ids"];
        tokens                      = _token_ids;
        return true;
    }

    std::vector<int> Encode(std::string input, ImageInfo img_info) override
    {
        std::vector<int> output;
        Encode(input, output, img_info);
        return output;
    }

    std::string Decode(const std::vector<int> &input) override
    {
        int cnt             = 2;
        std::string out_str = "";
        while (cnt--) {
            nlohmann::json j;
            j["token_ids"] = input;
            j["uid"]       = uid;
            auto ret       = cli->Post("/decode", j.dump(), "application/json");
            auto rep       = ret.value();
            if (rep.status != 200) {
                ALOGE("decode failed, status: %d, try again", rep.status);
                ALOGE("%s", rep.body.c_str());
                usleep(1000 * 1000);
                continue;
            }
            try {
                nlohmann::json j2 = nlohmann::json::parse(rep.body);
                out_str           = j2["text"];
                break;
            } catch (const std::exception &e) {
                ALOGE("json parse failed: %s, try again", e.what());
                ALOGE("%s", rep.body.c_str());
                usleep(1000 * 1000);
                continue;
            }
        }
        return out_str;
    }

    int GetBosID() override
    {
        return bos_id;
    }

    int GetEosID() override
    {
        return eos_id;
    }

    std::string apply_chat_template() override
    {
        std::ostringstream oss_prompt;
        int messages_len = messages_.size();
        for (auto &message : messages_) {
            messages_len--;
            switch (message.first) {
                case ROLE_USER: {
                    oss_prompt << message.second;
                } break;
                case ROLE_SYSTEM: {
                } break;
                case ROLE_ASSISTANT: {
                } break;
                case ROLE_ASSISTANT_HELP: {
                } break;
                default:
                    break;
            }
        }
        return oss_prompt.str();
    }
};

class ResultObj {
private:
    /* data */
public:
    bool success;
    nlohmann::json result;
    ResultObj(bool _success, const nlohmann::json &_result)
    {
        success = _success;
        result  = _result;
    }
};

class Tokenizer_Auto : public BaseTokenizer {
    bool _b_bos, _b_eos;
    std::string base_url;
    int bos_id, eos_id;
    pid_t pid       = 0;
    int pipe_c2p[2] = {0};  // C++ -> Python
    int pipe_p2c[2] = {0};  // Python -> C++
private:
    ResultObj readPython(int id, int timeout = 10000)
    {
        std::string message;
        message.reserve(1056);
        static char buffer[1056];
        while (timeout > 0) {
            ssize_t bytesRead = read(pipe_p2c[0], buffer, sizeof(buffer) - 1);
            if (bytesRead > 0) {
                for (int i = 0; i < bytesRead; i++) {
                    message += buffer[i];
                    if (buffer[i] == '\n') {
                        ALOGI("readPython:%s", message.c_str());
                        int pos = message.size() - 12;
                        if ((message.size() > 12) && (message.substr(pos) == "rpccontinue\n")) {
                            message.erase(pos);
                            continue;
                        }
                        nlohmann::json j2;
                        try {
                            j2 = nlohmann::json::parse(message);
                            if (j2["id"].is_number_integer() && j2["id"] == id) {
                                return ResultObj(true, j2["result"]);
                            } else if (j2["id"].is_null()) {
                                std::string errormesg;
                                if (j2["result"].is_string()) {
                                    errormesg = j2["result"];
                                }
                                ALOGE("python runtime error: %s", errormesg.c_str());
                                return ResultObj(false, j2);
                            }
                        } catch (const nlohmann::json::parse_error &e) {
                            ALOGE("JSON parse error: %s", e.what());
                            return ResultObj(false, "");
                        } catch (const nlohmann::json::type_error &e) {
                            ALOGE("JSON type error: %s", e.what());
                            return ResultObj(false, "");
                        } catch (const nlohmann::json::out_of_range &e) {
                            ALOGE("JSON out of range error: %s", e.what());
                            return ResultObj(false, "");
                        } catch (const std::exception &e) {
                            ALOGE("Standard exception: %s", e.what());
                            return ResultObj(false, "");
                        } catch (...) {
                            ALOGE("Unknown error occurred while parsing JSON");
                            return ResultObj(false, "");
                        }
                    }
                }
            } else {
                break;
            }
            usleep(10000);
            timeout -= 10;
        }
        return ResultObj(false, nullptr);
    }

    ResultObj callPython(nlohmann::json &rpcobj)
    {
        static int id        = 0;
        rpcobj["jsonrpc"]    = "2.0";
        rpcobj["id"]         = ++id;
        std::string _rawdata = rpcobj.dump(-1, ' ', true);
        _rawdata += "\n";
        std::string rawdata;
        rawdata.reserve(_rawdata.size());
        for (int i = 0; i < _rawdata.length(); i++) {
            if ((_rawdata.length() - i > 4) && (_rawdata[i] == '\\') && (_rawdata[i + 1] == '\\') &&
                (_rawdata[i + 2] == 'u')) {
                rawdata += '\\';
                rawdata += 'u';
                i += 2;
            } else {
                rawdata += _rawdata[i];
            }
        }
        const char *delstr = "\"DELSELF\":0";
        size_t pos         = rawdata.find(delstr);
        if (pos != std::string::npos) {
            rawdata.erase(pos, strlen(delstr));
        }
        int start_pos = 0;
        for (;;) {
            if (rawdata.length() - start_pos > 1024) {
                // ALOGE("callPython write:%.1024s", rawdata.c_str() + start_pos);
                write(pipe_c2p[1], rawdata.c_str() + start_pos, 1024);
                write(pipe_c2p[1], "rpccontinue\n", 12);
                start_pos += 1024;
                continue;
            }
            // ALOGE("callPython write:%.*s", rawdata.length() - start_pos, rawdata.c_str() + start_pos);
            write(pipe_c2p[1], rawdata.c_str() + start_pos, rawdata.length() - start_pos);
            break;
        }
        ALOGI("callPython write:%s", rawdata.c_str());
        return readPython(id);
    }

public:
    ~Tokenizer_Auto()
    {
        if (pid > 0) {
            kill(pid, SIGTERM);
            waitpid(pid, nullptr, 0);
        }
    }

    bool Init(std::string model_path = "tokenizer", bool b_bos = true, bool b_eos = false) override
    {
        // ALOGE("Tokenizer_Auto model_path:%s", model_path.c_str());
        std::string tokenizer_script = model_path;
        std::string model_id         = model_path.substr(0, model_path.find("_"));
        struct stat st;
        if (stat(tokenizer_script.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                tokenizer_script = "/opt/m5stack/scripts/llm-llm_tokenizer_auto.py";
            }
        } else {
            return false;
        }
        if (pipe(pipe_c2p) == -1 || pipe(pipe_p2c) == -1) {
            perror("pipe");
            return false;
        }
        pid = fork();
        if (pid == -1) {
            perror("fork");
            return false;
        }
        if (pid == 0) {
            close(pipe_c2p[1]);
            close(pipe_p2c[0]);
            dup2(pipe_c2p[0], STDIN_FILENO);
            dup2(pipe_p2c[1], STDOUT_FILENO);
            close(pipe_c2p[0]);
            close(pipe_p2c[1]);
            execlp("/usr/bin/python3", "python3", tokenizer_script.c_str(), "--model_id", model_id.c_str(), nullptr);
            perror("execlp");
            exit(EXIT_FAILURE);
        }
        close(pipe_c2p[0]);
        close(pipe_p2c[1]);
        auto ret = readPython(0, 15000);
        if (ret.success) {
            bos_id       = ret.result["bos_id"].is_number_integer() ? (int)ret.result["bos_id"] : 0;
            eos_id       = ret.result["eos_id"].is_number_integer() ? (int)ret.result["eos_id"] : 0;
            this->_b_bos = b_bos;
            this->_b_eos = b_eos;
        }
        return ret.success;
    }

    bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) override
    {
        nlohmann::json rpcobj;
        rpcobj["method"] = "encode";
        rpcobj["params"] = nlohmann::json::array({nlohmann::json::array({input}), {{"DELSELF", 0}}});
        auto ret         = callPython(rpcobj);
        if (ret.success) {
            if (ret.result.is_array()) {
                output = ret.result.get<std::vector<int>>();
            }
        }
        return ret.success;
    }

    std::vector<int> Encode(std::string input, ImageInfo img_info) override
    {
        std::vector<int> output;
        Encode(input, output, img_info);
        return output;
    }

    std::string Decode(const std::vector<int> &input) override
    {
        nlohmann::json rpcobj;
        rpcobj["method"] = "decode";
        rpcobj["params"] = nlohmann::json::array({nlohmann::json::array({input}), {{"DELSELF", 0}}});
        auto ret         = callPython(rpcobj);
        if (ret.success) {
            return ret.result.get<std::string>();
        }
        return "";
    }

    int GetBosID() override
    {
        return bos_id;
    }

    int GetEosID() override
    {
        return eos_id;
    }
    std::string apply_chat_template() override
    {
        nlohmann::json messages_list = nlohmann::json::array();
        for (auto &message : messages_) {
            nlohmann::json msg;
            switch (message.first) {
                case ROLE_USER: {
                    // {"role": "user", "content": prompt}
                    msg["role"]    = "user";
                    msg["content"] = (std::string)message.second;
                    messages_list.push_back(msg);
                } break;
                case ROLE_SYSTEM: {
                    msg["role"]    = "system";
                    msg["content"] = message.second;
                    messages_list.push_back(msg);
                } break;
                case ROLE_TOOL: {
                    msg["role"]    = "tool";
                    msg["content"] = message.second;
                    messages_list.push_back(msg);
                } break;
                case ROLE_IPYTHON: {
                    msg["role"]    = "ipython";
                    msg["content"] = message.second;
                    messages_list.push_back(msg);
                } break;
                case ROLE_ASSISTANT: {
                    msg["role"]    = "assistant";
                    msg["content"] = message.second;
                    messages_list.push_back(msg);
                } break;
                case ROLE_ASSISTANT_HELP: {
                } break;
                default:
                    break;
            }
        }
        nlohmann::json rpcobj;
        rpcobj["method"] = "apply_chat_template";
        rpcobj["params"] = nlohmann::json::array(
            {nlohmann::json::array({messages_list}), {{"tokenize", false}, {"add_generation_prompt", true}}});
        auto ret = callPython(rpcobj);
        if (ret.success) {
            std::string out_str = (std::string)ret.result;
            // ALOGE("out_str:%s", out_str.c_str());
            return out_str;
        }
        return "";
    }
};

std::shared_ptr<BaseTokenizer> CreateTokenizer(TokenizerType type)
{
    switch (type) {
        case TKT_LLaMa:
            return std::make_shared<TokenizerLLaMa>();
        case TKT_MINICPM:
            return std::make_shared<TokenizerMINICPM>();
        case TKT_HTTP:
            return std::make_shared<Tokenizer_Http>();
        case TKT_Qwen:
            return std::make_shared<TokenizerQwen>();
        case TKT_Phi3:
            return std::make_shared<TokenizerPhi3>();
        case TKT_AUTO:
            return std::make_shared<Tokenizer_Auto>();
        default:
            return nullptr;
    }
}
