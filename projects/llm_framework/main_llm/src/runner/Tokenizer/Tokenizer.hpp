#pragma once
#include <string>
#include <vector>
#include <memory>

enum TokenizerType { TKT_LLaMa, TKT_Qwen, TKT_HTTP, TKT_Phi3, TKT_END };

struct ImageInfo {
    int imgsz       = 448;
    int num_img     = 1;
    bool img_prompt = false;
};

class BaseTokenizer {
public:
    virtual bool Init(std::string model_path) = 0;

    virtual bool Reset(std::string system_prompt, std::vector<int> &tokens) = 0;

    virtual bool Encode(std::string input, std::string last_reply, std::vector<int> &tokens,
                        std::vector<int> &tokens_diff, ImageInfo img_info) = 0;

    virtual std::string Decode(const std::vector<int> &input) = 0;
    virtual int GetBosID()                                    = 0;
    virtual int GetEosID()                                    = 0;

    virtual bool isEnd(int id)
    {
        return id == GetEosID();
    }
};

std::shared_ptr<BaseTokenizer> CreateTokenizer(TokenizerType type);