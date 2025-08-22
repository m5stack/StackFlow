#pragma once
#include <string>
#include <vector>
#include <memory>
#include <list>
#include <utility>
#include <iostream>
enum TokenizerType { TKT_LLaMa, TKT_Qwen, TKT_HTTP, TKT_Phi3, TKT_MINICPM, TKT_AUTO, TKT_END };

enum TokenizeRole {
    ROLE_USER,           // 用户输入
    ROLE_SYSTEM,         // 提示词
    ROLE_TOOL,           // 工具
    ROLE_IPYTHON,        // 工具
    ROLE_ASSISTANT,      // 助手回复
    ROLE_ASSISTANT_HELP  // 询问句
};

struct ImageInfo {
    int imgsz       = 448;
    int num_img     = 1;
    bool img_prompt = false;
};

class BaseTokenizer {
public:
    std::list<std::pair<TokenizeRole, std::string>> messages_;

    void messages_clean()
    {
        messages_.clear();
    }

    virtual ~BaseTokenizer() = default;

    virtual bool Init(std::string model_path)
    {
        return false;
    };

    virtual bool Init(std::string model_path, bool b_bos, bool b_eos)
    {
        return false;
    };

    virtual bool Reset(std::string system_prompt, std::vector<int> &tokens)
    {
        return false;
    };

    virtual bool Encode(std::string input, std::string last_reply, std::vector<int> &tokens,
                        std::vector<int> &tokens_diff, ImageInfo img_info)
    {
        return false;
    };

    virtual bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) = 0;
    virtual std::vector<int> Encode(std::string input, ImageInfo img_info)               = 0;
    virtual std::string Decode(const std::vector<int> &input)                            = 0;
    virtual int GetBosID()                                                               = 0;
    virtual int GetEosID()                                                               = 0;

    virtual std::string apply_chat_template() = 0;

    virtual std::string messages_complete(TokenizeRole role, const std::string &content = "")
    {
        messages_.push_back(std::make_pair(role, content));
        if (ROLE_ASSISTANT_HELP == role)
            return apply_chat_template();
        else
            return "";
    }

    virtual bool isEnd(int id)
    {
        return id == GetEosID();
    }
};

std::shared_ptr<BaseTokenizer> CreateTokenizer(TokenizerType type);