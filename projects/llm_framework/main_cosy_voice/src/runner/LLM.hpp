#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "ax_sys_api.h"
#include "utils/sampling.hpp"
#include "utils/utils.hpp"

using SpeechToken = int;
// The container for speech tokens. std::deque is efficient for front/back operations.
using TokenBuffer = std::deque<SpeechToken>;

typedef std::function<void(int *, int, const char *, float, void *)> LLMRuningCallback;
// typedef void (*LLMRuningCallback)(int *p_token, int n_token, float token_per_sec, void *reserve);

struct LLMAttrType {
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num                       = 22;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";
    std::string filename_decoder_axmodel;
    std::string token2wav_axmodel_dir;
    std::string prompt_files;

    int prefill_token_num     = 96;  // auto calc
    int prefill_max_token_num = 512;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int precompute_len = 0;
    int prefill_grpid  = -1;

    TokenizerType tokenizer_type         = TKT_HTTP;
    std::string filename_tokenizer_model = "http://127.0.0.1:12345";
    std::string url_tokenizer_model      = "http://127.0.0.1:12345";
    bool b_bos = false, b_eos = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    std::string filename_llm_embed    = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    std::string filename_speech_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num              = 151936;
    int tokens_embed_size             = 896;

    int llm_embed_num     = 2;
    int llm_embed_size    = 896;
    int speech_embed_num  = 6564;
    int speech_embed_size = 896;

    int max_token_len = 127;  // auto calc

    int kv_cache_num  = 1024;  // auto calc
    int kv_cache_size = 256;   // auto calc

    bool b_use_mmap_load_embed        = false;
    bool b_dynamic_load_axmodel_layer = false;

    bool b_use_mmap_load_layer = true;

    bool b_use_topk = false;

    int n_timesteps;

    // bool b_live_print = true;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve                     = nullptr;
};

class LLM {
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;
    LLaMaEmbedSelector llm_embed_selector;
    LLaMaEmbedSelector speech_embed_selector;

    LLMAttrType _attr;

    struct LLMLayer {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;
    ax_runner_ax650 llm_decoder;

    // int prefill_grpid = 1;
    int decode_grpid = 0;
    bool b_stop      = false;
    int min_len      = -1;
    int max_len      = -1;

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer   = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.filename_tokenizer_model, attr.b_bos, attr.b_eos)) {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.filename_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size,
                                 attr.b_use_mmap_load_embed)) {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num,
                  attr.tokens_embed_size);
            return false;
        }
        if (!llm_embed_selector.Init(attr.filename_llm_embed, attr.llm_embed_num, attr.llm_embed_size,
                                     attr.b_use_mmap_load_embed)) {
            ALOGE("llm_embed_selector.Init(%s, %d, %d) failed", attr.filename_llm_embed.c_str(), attr.llm_embed_num,
                  attr.llm_embed_size);
            return false;
        }
        if (!speech_embed_selector.Init(attr.filename_speech_embed, attr.speech_embed_num, attr.speech_embed_size,
                                        attr.b_use_mmap_load_embed)) {
            ALOGE("speech_embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(),
                  attr.speech_embed_num, attr.speech_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");

        llama_layers.resize(attr.axmodel_num);
        // prefill_layers.resize(attr.prefill_axmodel_num);
        ALOGI("attr.axmodel_num:%d", attr.axmodel_num);
        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++) {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            if (!attr.b_dynamic_load_axmodel_layer) {
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), false);
                if (ret != 0) {
                    ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                    return false;
                }
                int remain_cmm = get_remaining_cmm_size();
                sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            } else {
                if (!attr.b_use_mmap_load_layer) {
                    if (!read_file(llama_layers[i].filename, llama_layers[i].layer_buffer_vec)) {
                        ALOGE("read_file(%s) failed", llama_layers[i].filename.c_str());
                        return false;
                    }
                } else {
                    llama_layers[i].layer_buffer.open_file(llama_layers[i].filename.c_str());
                }

                sprintf(axmodel_path, "read_file %s ok", llama_layers[i].filename.c_str());
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
        if (ret != 0) {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        ret = llm_decoder.init(attr.filename_decoder_axmodel.c_str(), false);
        if (ret != 0) {
            ALOGE("init llm decoder axmodel(%s) failed", attr.filename_decoder_axmodel.c_str());
            return false;
        }

        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        if (attr.b_dynamic_load_axmodel_layer) {
            // 加载第一层获取shape信息
            auto &layer = llama_layers[0];
            int ret;
            if (_attr.b_use_mmap_load_layer) {
                ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size());
            } else {
                ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size());
            }
            if (ret != 0) {
                ALOGE("init axmodel(%s) failed", layer.filename.c_str());
            }
        }

        {
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            printf("\n");
            ALOGI("max_token_len : %d", _attr.max_token_len);
            // auto &input_k_cache = llama_layers[0].layer.get_input("K_cache");
            // auto &output_k_cache_out = llama_layers[0].layer.get_output("K_cache_out");
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num =
                llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num) {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer.get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
            for (size_t i = 0; i < llama_layers[0].layer.get_num_input_groups() - 1; i++) {
                int prefill_max_kv_cache_num = llama_layers[0].layer.get_input(i + 1, "K_cache").vShape[1];
                ALOGI("grp: %ld, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num =
                _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }
        if (attr.b_dynamic_load_axmodel_layer) {
            for (int i = 0; i < attr.axmodel_num; i++) {
                auto &layer = llama_layers[i];
                layer.layer.deinit();
            }
        }

        // Reset();
        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++) {
            llama_layers[i].layer.release();
        }
        llama_post.release();
        llm_decoder.release();
        embed_selector.Deinit();
        llm_embed_selector.Deinit();
        speech_embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int TextToken2Embeds(std::vector<int> &token_ids, std::vector<unsigned short> &token_embeds)
    {
        if (token_embeds.empty() || token_embeds.size() < token_ids.size() * _attr.tokens_embed_size) {
            token_embeds.resize(token_ids.size() * _attr.tokens_embed_size);
        }

        for (size_t i = 0; i < token_ids.size(); i++) {
            embed_selector.getByIndex(token_ids[i], token_embeds.data() + i * _attr.tokens_embed_size);
        }
        return token_embeds.size();
    }

    int SpeechToken2Embeds(std::vector<int> &token_ids, std::vector<unsigned short> &token_embeds)
    {
        if (token_embeds.empty() || token_embeds.size() < token_ids.size() * _attr.speech_embed_size) {
            token_embeds.resize(token_ids.size() * _attr.speech_embed_size);
        }

        for (size_t i = 0; i < token_ids.size(); i++) {
            speech_embed_selector.getByIndex(token_ids[i], token_embeds.data() + i * _attr.speech_embed_size);
        }
        return token_embeds.size();
    }

    int Encode(std::vector<unsigned short> &out_embed, std::vector<std::vector<int>> &position_ids, std::string text,
               const std::vector<unsigned short> &prompt_text_embeds, const std::vector<unsigned short> &prompt_speech_embeds)
    {
        // std::vector<int> prompt_ids = tokenizer->Encode(prompt_text, true);
        ImageInfo img_info;
        img_info.img_prompt       = false;
        std::vector<int> text_ids = tokenizer->Encode(text, img_info);
        int prompt_ids_size       = prompt_text_embeds.size() / _attr.tokens_embed_size;
        int total_size = prompt_ids_size + text_ids.size() + 2 + prompt_speech_embeds.size() / _attr.speech_embed_size;
        if (total_size > _attr.prefill_max_token_num) {
            ALOGE("input embeding size(%d) > prefill_max_token_num(%d)", total_size, _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(total_size * _attr.tokens_embed_size);

        llm_embed_selector.getByIndex(0, out_embed.data() + 0 * _attr.tokens_embed_size);

        // for (size_t i = 0; i < prompt_ids_size; i++)
        // {
        //     embed_selector.getByIndex(prompt_ids[i], out_embed.data() + (1+i) * _attr.tokens_embed_size);
        // }
        memcpy(out_embed.data() + _attr.tokens_embed_size, prompt_text_embeds.data(),
               prompt_text_embeds.size() * sizeof(unsigned short));

        for (size_t i = 0; i < text_ids.size(); i++) {
            embed_selector.getByIndex(text_ids[i],
                                      out_embed.data() + (1 + prompt_ids_size + i) * _attr.tokens_embed_size);
        }

        llm_embed_selector.getByIndex(
            1, out_embed.data() + (1 + prompt_ids_size + text_ids.size()) * _attr.tokens_embed_size);

        // for (size_t i = 0; i < prompt_speech_tokens.size(); i++)
        // {
        //     speech_embed_selector.getByIndex(prompt_speech_tokens[i], out_embed.data() +
        //     (1+prompt_ids_size+text_ids.size()+1) * _attr.tokens_embed_size);
        // }

        memcpy(out_embed.data() + (1 + prompt_ids_size + text_ids.size() + 1) * _attr.tokens_embed_size,
               prompt_speech_embeds.data(), prompt_speech_embeds.size() * sizeof(unsigned short));

        std::vector<int> pos_ids;
        for (size_t i = 0; i < total_size; i++) {
            pos_ids.push_back(i);
        }
        position_ids.push_back(pos_ids);

        min_len = text_ids.size() * 2;
        max_len = text_ids.size() * 20;
        return 0;
    }

    int Run(const std::string &input_str, const std::vector<unsigned short> &prompt_text_embeds,
            const std::vector<unsigned short> &prompt_speech_embeds, TokenBuffer &token_buffer,
            std::mutex &buffer_mutex, std::condition_variable &buffer_cv, std::atomic<bool> &llm_finished)
    {
        std::vector<unsigned short> text_embed;
        std::vector<std::vector<int>> position_ids;
        Encode(text_embed, position_ids, input_str, prompt_text_embeds, prompt_speech_embeds);
        return Run(text_embed, position_ids, token_buffer, buffer_mutex, buffer_cv, llm_finished);
    }

    int Run(std::vector<unsigned short> &text_embed, std::vector<std::vector<int>> &position_ids,
            TokenBuffer &token_buffer, std::mutex &buffer_mutex, std::condition_variable &buffer_cv,
            std::atomic<bool> &llm_finished)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.speech_embed_size, 0);

        std::vector<int> cached_token;
        std::vector<int> token_ids;

        // std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
        int input_embed_num   = text_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        if (input_embed_num > _attr.prefill_max_token_num) {
            ALOGE("input token num(%d) > prefill_max_token_num(%d)", input_embed_num, _attr.prefill_max_token_num);
            return -1;
        }

        int kv_cache_num;
        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++) {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        int max_pos_id = 0;
        for (size_t p = 0; p < prefill_split_num; p++) {
            if (b_stop) {
                break;
            }
            _attr.prefill_grpid = p + 1;
            kv_cache_num        = p * _attr.prefill_token_num;
            std::vector<unsigned short> mask_tmp;
            mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1) {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            ALOGI("input_num_token:%d", input_num_token);
            for (size_t i = 0; i < _attr.prefill_token_num; i++) {
                if (i < input_num_token) {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr          = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

                    for (int j = 0; j < _attr.precompute_len + p * _attr.prefill_token_num; j++) {
                        mask_ptr[j] = 0;
                    }

                    for (int j = mask_current_start; j < mask_current_start + i + 1; j++) {
                        mask_ptr[j] = 0;
                    }
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            if (p == (prefill_split_num - 1)) {
                memcpy(
                    embed_tmp.data(), text_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                    (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            } else {
                memcpy(embed_tmp.data(), text_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                       _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++) {
                if (b_stop) {
                    break;
                }

                auto &layer = llama_layers[m];

                if (_attr.b_dynamic_load_axmodel_layer) {
                    int ret;
                    if (_attr.b_use_mmap_load_layer) {
                        ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size());
                    } else {
                        ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size());
                    }
                    if (ret != 0) {
                        ALOGE("init axmodel(%s) failed", layer.filename.c_str());
                    }
                }

                // set indices
                auto &input_indices             = layer.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                // ALOGI("position_ids");
                for (unsigned int i = 0; i < position_ids.size(); i++) {
                    for (unsigned int j = _attr.precompute_len + p * _attr.prefill_token_num, jj = 0;
                         j < _attr.precompute_len + (p + 1) * _attr.prefill_token_num; j++, jj++) {
                        if (j < position_ids[i].size()) {
                            input_indices_ptr[i * _attr.prefill_token_num + jj] = position_ids[i][j];
                            if (position_ids[i][j] > max_pos_id) {
                                max_pos_id = position_ids[i][j];
                            }
                        }
                    }
                }

                // set mask
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));
                // set input
                auto &input_input = layer.layer.get_input(_attr.prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(_attr.prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (_attr.precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset, (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset, (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                for (int gid = _attr.prefill_grpid + 1; gid < prefill_split_num + 1; gid++) {
                    auto &input_prefill_k_cache = layer.layer.get_input(gid, "K_cache");
                    memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                           (void *)output_k_cache.pVirAddr,
                           sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);
                }

                for (int gid = _attr.prefill_grpid + 1; gid < prefill_split_num + 1; gid++) {
                    auto &input_prefill_v_cache = layer.layer.get_input(gid, "V_cache");
                    memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                           (void *)output_v_cache.pVirAddr,
                           sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);
                }

                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));

                if (_attr.b_dynamic_load_axmodel_layer) {
                    layer.layer.deinit();
                }
            }
            if (p == (prefill_split_num - 1)) {
                memcpy(embed.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }

        int next_token = -1;
        t_cqdm cqdm    = create_cqdm(_attr.max_token_len, 32);
        int max_index;
        std::vector<float> scores(_attr.speech_embed_num, 0.0f);
        {
            // post process
            auto &input = llama_post.get_input(0);
            memcpy((void *)input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();
            if (_attr.b_use_topk) {
                AX_SYS_MinvalidateCache(llama_post.get_output("indices").phyAddr,
                                        llama_post.get_output("indices").pVirAddr,
                                        llama_post.get_output("indices").nSize);
                max_index = *(int *)llama_post.get_output("indices").pVirAddr;
            } else {
                auto &output_post        = llama_post.get_output("output_norm");  // 1 means get rmsnorm output
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                std::vector<float> logits(output_post.nSize / sizeof(unsigned short));
                for (int i = 0; i < output_post.nSize / sizeof(unsigned short); i++) {
                    unsigned int proc = post_out[i] << 16;
                    logits[i]         = *reinterpret_cast<float *>(&proc);
                }

                auto &input_decoder = llm_decoder.get_input(0);
                memcpy(input_decoder.pVirAddr, logits.data(), logits.size() * sizeof(float));
                llm_decoder.inference();

                auto &output_decoder = llm_decoder.get_output(0);
                float *post_decoder  = (float *)output_decoder.pVirAddr;

                memcpy(scores.data(), post_decoder, output_decoder.nSize);
                max_index = sampling::sampling_ids(scores, cached_token, _attr.speech_embed_num - 3, true);
            }
            next_token = max_index;

            if (max_index >= _attr.speech_embed_num - 3) {
                llm_finished = true;
                buffer_cv.notify_all();
                ALOGI("hit eos, llm finished");
                return -1;
            }

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                token_buffer.push_back(max_index);
            }
            buffer_cv.notify_one();
            // ALOGI("token_buffer push %d", max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;

        for (unsigned int indices = max_pos_id + 1; indices - max_pos_id < max_len; indices++) {
            if (b_stop) {
                break;
            }

            speech_embed_selector.getByIndex(next_token, embed.data());
            memcpy((void *)llama_layers[0].layer.get_input(decode_grpid, "input").pVirAddr, embed.data(),
                   llama_layers[0].layer.get_input(decode_grpid, "input").nSize);

            {
                std::vector<float> float_embeds(embed.size());
                for (int i = 0; i < embed.size(); i++) {
                    unsigned int proc = embed[i] << 16;
                    float_embeds[i]   = *reinterpret_cast<float *>(&proc);
                }
            }

            for (int m = 0; m < _attr.axmodel_num; m++) {
                if (b_stop) {
                    break;
                }

                auto &layer = llama_layers[m];

                if (_attr.b_dynamic_load_axmodel_layer) {
                    int ret;
                    if (_attr.b_use_mmap_load_layer) {
                        ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size());
                    } else {
                        ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size());
                    }
                    if (ret != 0) {
                        ALOGE("init axmodel(%s) failed", layer.filename.c_str());
                    }
                }

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy((void *)input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                memcpy((unsigned short *)input_k_cache.pVirAddr + indices * _attr.kv_cache_size,
                       (void *)output_k_cache.pVirAddr, output_k_cache.nSize);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                memcpy((unsigned short *)input_v_cache.pVirAddr + indices * _attr.kv_cache_size,
                       (void *)output_v_cache.pVirAddr, output_v_cache.nSize);

                if (m == _attr.axmodel_num - 1) {
                    memcpy((void *)llama_post.get_input(0).pVirAddr,
                           (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr,
                           llama_post.get_input(0).nSize);
                } else if (m < _attr.axmodel_num - 1) {
                    memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").pVirAddr,
                           (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr,
                           layer.layer.get_input(decode_grpid, "input").nSize);
                }
            }

            mask[indices] = 0;
            {
                llama_post.inference();

                auto &output_post        = llama_post.get_output("output_norm");  // 1 means get rmsnorm output
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                std::vector<float> logits(output_post.nSize / sizeof(unsigned short));
                for (int i = 0; i < output_post.nSize / sizeof(unsigned short); i++) {
                    unsigned int proc = post_out[i] << 16;
                    logits[i]         = *reinterpret_cast<float *>(&proc);
                }

                auto &input_decoder = llm_decoder.get_input(0);
                memcpy(input_decoder.pVirAddr, logits.data(), logits.size() * sizeof(float));
                llm_decoder.inference();

                auto &output_decoder = llm_decoder.get_output(0);
                float *post_decoder  = (float *)output_decoder.pVirAddr;

                memcpy(scores.data(), post_decoder, output_decoder.nSize);

                bool ignore_eos = false;
                if (indices < min_len) {
                    ignore_eos = true;
                }

                max_index  = sampling::sampling_ids(scores, cached_token, _attr.speech_embed_num - 3, ignore_eos);
                next_token = max_index;

                if (max_index == _attr.speech_embed_num - 3) {
                    b_hit_eos    = true;
                    llm_finished = true;
                    buffer_cv.notify_all();
                    ALOGI("hit eos, llm finished");
                    break;
                }

                if (max_index < _attr.speech_embed_num - 3) {
                    token_ids.push_back(max_index);
                    cached_token.push_back(max_index);
                    {
                        std::lock_guard<std::mutex> lock(buffer_mutex);
                        token_buffer.push_back(max_index);
                    }
                    buffer_cv.notify_one();
                    // ALOGI("token_buffer push %d", max_index);
                }
            }

            if (b_hit_eos) {
                llm_finished = true;
                buffer_cv.notify_all();
                ALOGI("hit eos, llm finished");
                break;
            }
        }

        llm_finished = true;
        buffer_cv.notify_all();
        ALOGI("llm finished");

        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGI("total decode tokens:%d", cached_token.size());
        ALOGN("hit eos, decode avg %.2f token/s\n", cached_token.size() / (t_cost_ms / 1000));

        for (size_t i = 0; i < _attr.axmodel_num; i++) {
            for (size_t j = 0; j < llama_layers[i].layer.get_num_input_groups(); j++) {
                memset((void *)llama_layers[i].layer.get_input(j, "K_cache").pVirAddr, 0,
                       llama_layers[i].layer.get_input(j, "K_cache").nSize);
                memset((void *)llama_layers[i].layer.get_input(j, "V_cache").pVirAddr, 0,
                       llama_layers[i].layer.get_input(j, "V_cache").nSize);
            }
        }

        return cached_token.size();
    }
};