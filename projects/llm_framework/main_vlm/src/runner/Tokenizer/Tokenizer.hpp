#pragma once
#include <string>
#include <vector>
#include <memory>

enum TokenizerType { TKT_LLaMa, TKT_Qwen, TKT_HTTP, TKT_Phi3, TKT_END };

enum class ImgType { None, Image, Video };

struct ImageInfo {
    int imgsz         = 448;
    int num_img       = 1;
    bool img_prompt   = false;
    int img_token_num = -1;
    ImgType type      = ImgType::None;
};

class BaseTokenizer {
public:
    virtual bool Init(std::string model_path)                                            = 0;
    virtual bool Init(std::string model_path, bool b_bos, bool b_eos)                    = 0;
    virtual bool Init_new(std::string model_path, bool b_bos, bool b_eos)                = 0;
    virtual bool Reset(std::string system_prompt, std::vector<int> &tokens)              = 0;
    virtual bool Encode(std::string input, std::string last_reply, std::vector<int> &tokens,
                        std::vector<int> &tokens_diff, ImageInfo img_info)               = 0;
    virtual bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) = 0;
    virtual std::vector<int> Encode(std::string input, ImageInfo img_info)               = 0;
    virtual std::vector<int> Encode_ctx(std::string input, ImageInfo img_info, std::vector<int> &tokens_ids,
                                        std::vector<int> &tokens_diff)                   = 0;
    virtual std::string Decode(const std::vector<int> input)                             = 0;
    virtual int GetBosID()                                                               = 0;
    virtual int GetEosID()                                                               = 0;
    virtual int GetImgStartID()                                                          = 0;
    virtual int GetImgContextID()                                                        = 0;

    virtual bool isEnd(int id)
    {
        return id == GetEosID();
    }
};

std::shared_ptr<BaseTokenizer> CreateTokenizer(TokenizerType type);