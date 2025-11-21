#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>

#include "bfloat16.hpp"
#include "image_processor.hpp"
#include "mrope.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "opencv2/opencv.hpp"
#include "ax_sys_api.h"
#include "LLMPostprocess.hpp"

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

typedef std::function<void(int *, int, const char *, float, void *)> LLMRuningCallback;

struct LLMAttrType {
    std::string system_prompt;

    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    std::string post_config_path          = "post_config.json";
    int axmodel_num                       = 22;

    std::string filename_image_encoder_axmodel  = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    std::string filename_vpm_encoder_axmodel    = "minicpmv/vpm_resampler_version0_fp16.axmodel";
    std::string filename_vpm_resampler_axmodedl = "minicpmv/vpm_resampler_version0_fp16.axmodel";

    int image_encoder_width       = 448;
    int image_encoder_height      = 448;
    int vpm_width                 = 280;
    int vpm_height                = 280;
    bool b_vpm_two_stage          = false;
    int IMAGE_CONTEXT_TOKEN       = 151667;
    int IMAGE_START_TOKEN         = 151665;
    int IMAGE_ENCODER_INPUT_NCHW  = -1;
    int IMAGE_ENCODER_OUTPUT_BF16 = -1;

    int prefill_token_num     = 96;
    int prefill_max_token_num = 512;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    TokenizerType tokenizer_type         = TKT_LLaMa;
    std::string filename_tokenizer_model = "tokenizer.model";
    std::string url_tokenizer_model;
    bool b_bos                        = false;
    bool b_eos                        = false;
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num              = 32000;
    int img_token_id                  = 151667;
    int tokens_embed_size             = 2048;

    int max_token_len = 127;
    int kv_cache_num  = 1024;
    int kv_cache_size = 256;

    int precompute_len = 0;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int prefill_grpid = -1;

    bool enable_temperature = false;
    float temperature       = 0.7f;

    bool enable_top_p_sampling = false;
    float top_p                = 0.7f;

    bool enable_top_k_sampling = true;
    int top_k                  = 10;

    bool enable_repetition_penalty = false;
    float repetition_penalty       = 1.2f;
    int penalty_window             = 50;

    bool b_use_mmap_load_embed        = false;
    bool b_dynamic_load_axmodel_layer = false;
    bool b_use_mmap_load_layer        = true;
    bool b_video                      = false;

    LLMRuningCallback runing_callback = nullptr;
    void *reserve                     = nullptr;
};

class LLM {
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

    LLMAttrType _attr;

    struct LLMLayer {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;

    ax_runner_ax650 vpm_encoder, vpm_resampler;

    int prefill_grpid = 1;
    int decode_grpid  = 0;

    bool b_stop = false;

    LLMPostprocess postprocess;
    static int post_process(LLMPostprocess &postprocess, unsigned short *p, int n, std::vector<int> &history,
                            float *val = 0)
    {
        std::vector<float> logits(n);
        for (int i = 0; i < n; i++) {
            unsigned int proc = p[i] << 16;
            logits[i]         = *reinterpret_cast<float *>(&proc);
        }

        return postprocess.apply(logits, history);
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 4, 32);
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
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");
        llama_layers.resize(attr.axmodel_num);

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
        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        if (_attr.b_vpm_two_stage) {
            ret = vpm_encoder.init(attr.filename_vpm_encoder_axmodel.c_str(), false);
            if (ret != 0) {
                ALOGE("init vpm axmodel(%s) failed", attr.filename_vpm_encoder_axmodel.c_str());
                return false;
            }

            ret = vpm_resampler.init(attr.filename_vpm_resampler_axmodedl.c_str(), false);
            if (ret != 0) {
                ALOGE("init vpm axmodel(%s) failed", attr.filename_vpm_resampler_axmodedl.c_str());
                return false;
            }

            _attr.vpm_height = vpm_encoder.get_input(0).vShape[1];
            _attr.vpm_width  = vpm_encoder.get_input(0).vShape[2];
        } else {
            ret = vpm_resampler.init(attr.filename_vpm_resampler_axmodedl.c_str(), false);
            if (ret != 0) {
                ALOGE("init vpm axmodel(%s) failed", attr.filename_vpm_resampler_axmodedl.c_str());
                return false;
            }
            _attr.vpm_height = vpm_resampler.get_input(0).vShape[1];
            _attr.vpm_width  = vpm_resampler.get_input(0).vShape[2];
        }

        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init vpm axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 3, "count", axmodel_path);

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
            int max_token_len   = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            _attr.max_token_len = max_token_len > _attr.max_token_len ? _attr.max_token_len : max_token_len;
            ALOGI("max_token_len : %d", _attr.max_token_len);
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num =
                llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num) {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer.get_input(prefill_grpid, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
            ALOGI("vpm_height : %d,vpm_width : %d", _attr.vpm_height, _attr.vpm_width);
        }
        if (attr.b_dynamic_load_axmodel_layer) {
            auto &layer = llama_layers[0];
            layer.layer.deinit();
        }
        nlohmann::json dynamic_config;

        dynamic_config["enable_temperature"] = _attr.enable_temperature;
        dynamic_config["temperature"]        = _attr.temperature;

        dynamic_config["enable_repetition_penalty"] = _attr.enable_repetition_penalty;
        dynamic_config["repetition_penalty"]        = _attr.repetition_penalty;
        dynamic_config["penalty_window"]            = _attr.penalty_window;

        dynamic_config["enable_top_p_sampling"] = _attr.enable_top_p_sampling;
        dynamic_config["top_p"]                 = _attr.top_p;

        dynamic_config["enable_top_k_sampling"] = _attr.enable_top_k_sampling;
        dynamic_config["top_k"]                 = _attr.top_k;

        if (!postprocess.load_config(attr.post_config_path)) {
            ALOGW("load postprocess config(%s) failed", attr.post_config_path.c_str());
        }

        if (!postprocess.load_config(dynamic_config)) {
            ALOGW("load postprocess config(%s) failed", dynamic_config.dump(4).c_str());
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
        vpm_encoder.release();
        vpm_resampler.release();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int Encode(cv::Mat src, std::vector<unsigned short> &out_embed)
    {
        timer t;
        t.start();
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(_attr.vpm_width, _attr.vpm_height));
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

        if (_attr.b_vpm_two_stage) {
            void *data = vpm_encoder.get_input(0).pVirAddr;
            memcpy(data, dst.data, dst.rows * dst.cols * 3);
            vpm_encoder.inference();
            AX_SYS_MinvalidateCache(vpm_encoder.get_output(0).phyAddr, vpm_encoder.get_output(0).pVirAddr,
                                    vpm_encoder.get_output(0).nSize);
            memcpy(vpm_resampler.get_input(0).pVirAddr, vpm_encoder.get_output(0).pVirAddr,
                   vpm_encoder.get_output(0).nSize);
        } else {
            void *data = vpm_resampler.get_input(0).pVirAddr;
            memcpy(data, dst.data, dst.rows * dst.cols * 3);
        }

        vpm_resampler.inference();
        out_embed.resize(vpm_resampler.get_output(0).nSize / sizeof(float));
        AX_SYS_MinvalidateCache(vpm_resampler.get_output(0).phyAddr, vpm_resampler.get_output(0).pVirAddr,
                                vpm_resampler.get_output(0).nSize);

        float *output_data = (float *)vpm_resampler.get_output(0).pVirAddr;
        for (size_t i = 0; i < out_embed.size(); i++) {
            out_embed[i] = bfloat16(output_data[i]).data;
        }

        ALOGI("image encode time : %f ms, size : %d", t.cost(), out_embed.size());
        return 0;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt = "What is in the image?")
    {
        ImageInfo img_info;
        img_info.img_prompt        = false;
        std::vector<int> input_ids = tokenizer->Encode(prompt, img_info);
        if (input_ids.size() > _attr.prefill_token_num) {
            ALOGE("input_ids(%d) > prefill_token_num(%d)", input_ids.size(), _attr.prefill_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++) {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        return 0;
    }

    int Encode(std::vector<unsigned short> &img_embed, std::vector<unsigned short> &out_embed,
               std::string prompt = "What is in the image?")
    {
        ImageInfo img_info;
        img_info.img_prompt        = true;
        img_info.num_img           = 1;
        img_info.imgsz             = _attr.image_encoder_width;
        std::vector<int> input_ids = tokenizer->Encode(prompt, img_info);

        // constexpr int img_token_id = 49190;	// smolvlm
        // constexpr int img_token_id = 151667; // InternVL2.5
        int offset            = 0;
        int img_context_count = 0;

        for (size_t i = 0; i < input_ids.size(); i++) {
            if (input_ids[i] == _attr.img_token_id) {
                img_context_count++;
                if (img_context_count == 1) {
                    offset = i;
                }
            }
        }

        if (offset == 0) {
            ALOGE("offset == 0");
            return -1;
        }

        if (img_context_count != img_embed.size() / _attr.tokens_embed_size) {
            ALOGE("img_context_count(%d) != img_embed.size() / tokens_embed_size(%d)", img_context_count,
                  img_embed.size() / _attr.tokens_embed_size);
            return -1;
        }

        if (input_ids.size() > _attr.prefill_token_num) {
            ALOGE("input_ids(%d) > prefill_token_num(%d)", input_ids.size(), _attr.prefill_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++) {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }
        memcpy(out_embed.data() + offset * _attr.tokens_embed_size, img_embed.data(),
               img_embed.size() * sizeof(unsigned short));

        return 0;
    }

    std::string Run(std::string input_str)
    {
        std::vector<unsigned short> test_embed;
        Encode(test_embed, input_str);
        return Run(test_embed);
    }

    std::string Run(std::vector<unsigned short> test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> mask_p(_attr.prefill_token_num * _attr.prefill_token_num, bf16.data);

        for (size_t i = 0; i < _attr.prefill_token_num; i++) {
            for (size_t j = 0; j < i + 1; j++) {
                mask_p[i * _attr.prefill_token_num + j] = 0;
            }
        }

        std::vector<int> cached_token;
        std::vector<int> token_ids;
        // std::vector<int> token_ids = tokenizer->Encode(input_str);
        // int len_of_input = token_ids.size();
        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        // ALOGI("input_embed_num(%d)", input_embed_num);

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++) {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        for (unsigned int m = 0; m < _attr.axmodel_num; m++) {
            if (b_stop) {
                break;
            }

            auto &layer       = llama_layers[m];
            auto &layer_llama = llama_layers[m];

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

            auto &input_indices             = layer.layer.get_input(prefill_grpid, "indices");
            unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
            for (unsigned int i = 0; i < input_embed_num; i++) {
                input_indices_ptr[i] = i;
            }

            auto &input_mask = layer.layer.get_input(prefill_grpid, "mask");
            memcpy(input_mask.pVirAddr, mask_p.data(), mask_p.size() * sizeof(unsigned short));

            auto &input_input = layer.layer.get_input(prefill_grpid, "input");
            memcpy(input_input.pVirAddr, test_embed.data(), test_embed.size() * sizeof(unsigned short));
            if (m == 0) {
                test_embed.resize(_attr.prefill_token_num * _attr.tokens_embed_size);
            }

            layer.layer.inference(prefill_grpid);

            auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
            AX_SYS_MinvalidateCache(output_k_cache.phyAddr, output_k_cache.pVirAddr, output_k_cache.nSize);
            auto &input_k_cache = layer_llama.layer.get_input(decode_grpid, "K_cache");
            memcpy(input_k_cache.pVirAddr, output_k_cache.pVirAddr,
                   sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

            auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");
            AX_SYS_MinvalidateCache(output_v_cache.phyAddr, output_v_cache.pVirAddr, output_v_cache.nSize);
            auto &input_v_cache = layer_llama.layer.get_input(decode_grpid, "V_cache");
            memcpy(input_v_cache.pVirAddr, output_v_cache.pVirAddr,
                   sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

            auto &output = layer.layer.get_output(prefill_grpid, "output");
            AX_SYS_MinvalidateCache(output.phyAddr, output.pVirAddr, output.nSize);
            memcpy(test_embed.data(), output.pVirAddr, test_embed.size() * sizeof(unsigned short));
            if (_attr.b_dynamic_load_axmodel_layer) {
                layer.layer.deinit();
            }
        }

        int next_token = -1;
        t_cqdm cqdm    = create_cqdm(_attr.max_token_len, 32);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);

        memcpy(embed.data(), test_embed.data() + (input_embed_num - 1) * _attr.tokens_embed_size,
               _attr.tokens_embed_size * sizeof(unsigned short));

        {
            // post process
            auto &input = llama_post.get_input("input");
            memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();

            int max_index;

            auto &output_post = llama_post.get_output("output");
            AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
            unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
            float max_val            = -MAXFLOAT;
            max_index                = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, &max_val);

            next_token = max_index;

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;
        for (unsigned int indices = input_embed_num; indices < _attr.max_token_len; indices++) {
            if (b_stop) {
                break;
            }

            embed_selector.getByIndex(next_token, embed);

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

                auto &input_k_cache               = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                // memcpy(input_k_cache.pVirAddr, k_caches[m].data(), sizeof(unsigned short) * k_caches[m].size());
                auto &input_v_cache               = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;
                // memcpy(input_v_cache.pVirAddr, v_caches[m].data(), sizeof(unsigned short) * v_caches[m].size());

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(decode_grpid, "input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                AX_SYS_MinvalidateCache(output_k_cache.phyAddr, output_k_cache.pVirAddr, output_k_cache.nSize);
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                AX_SYS_MinvalidateCache(output_v_cache.phyAddr, output_v_cache.pVirAddr, output_v_cache.nSize);
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(decode_grpid, "output");
                AX_SYS_MinvalidateCache(output.phyAddr, output.pVirAddr, output.nSize);
                memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));
                if (_attr.b_dynamic_load_axmodel_layer) {
                    layer.layer.deinit();
                }
            }

            mask[indices] = 0;
            {
                // post process
                auto &input = llama_post.get_input("input");
                memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                llama_post.inference();
                int max_index;

                auto &output_post = llama_post.get_output("output");
                AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val            = -MAXFLOAT;
                max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, &max_val);

                next_token = max_index;

                if (tokenizer->isEnd(max_index)) {
                    if (cached_token.size() && _attr.runing_callback) {
                        float t_cost_ms     = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out        = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec,
                                              _attr.reserve);
                        cached_token.clear();
                    }
                    b_hit_eos = true;
                    break;
                }
                token_ids.push_back(max_index);

                if (_attr.runing_callback) {
                    cached_token.push_back(max_index);
                    if (cached_token.size() >= 3) {
                        float t_cost_ms     = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out        = tokenizer->Decode(cached_token);
                        if (!tmp_out.empty() && tmp_out.back() != 0xBD) {
                            _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(),
                                                  token_per_sec, _attr.reserve);
                            cached_token.clear();
                        }
                    }
                }
            }

            if (_attr.runing_callback == nullptr) update_cqdm(&cqdm, indices, "token", "");
            if (b_hit_eos) {
                break;
            }
        }
        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_cost_ms / 1000));

        final_out = tokenizer->Decode(token_ids);

        return final_out;
    }
};

class LLM_CTX {
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;
    std::vector<std::vector<unsigned short>> imgs_embed_;

    LLMAttrType _attr;

    struct LLMLayer {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;
    ax_runner_ax650 image_encoder;

    //
    int decode_grpid = 0;

    bool b_stop = false;

    LLMPostprocess postprocess;
    static int post_process(LLMPostprocess &postprocess, unsigned short *p, int n, std::vector<int> &history,
                            float *val = 0)
    {
        std::vector<float> logits(n);
        for (int i = 0; i < n; i++) {
            unsigned int proc = p[i] << 16;
            logits[i]         = *reinterpret_cast<float *>(&proc);
        }

        return postprocess.apply(logits, history);
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer   = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.url_tokenizer_model)) {
            ALOGE("tokenizer.Init(%s) failed", attr.url_tokenizer_model.c_str());
            return false;
        }
        std::vector<int> _token_ids;
        tokenizer->Reset(attr.system_prompt, _token_ids);
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size,
                                 attr.b_use_mmap_load_embed)) {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num,
                  attr.tokens_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");

        llama_layers.resize(attr.axmodel_num);

        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++) {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), false);
            if (ret != 0) {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 2, "count", axmodel_path);
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
        if (ret != 0) {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        ret = image_encoder.init(attr.filename_image_encoder_axmodel.c_str());
        if (ret != 0) {
            ALOGE("init vpm axmodel(%s) failed", attr.filename_image_encoder_axmodel.c_str());
            return false;
        }

        _attr.IMAGE_CONTEXT_TOKEN = tokenizer->GetImgContextID();
        _attr.IMAGE_START_TOKEN   = tokenizer->GetImgStartID();

        ALOGI("IMAGE_CONTEXT_TOKEN: %d, IMAGE_START_TOKEN: %d", _attr.IMAGE_CONTEXT_TOKEN, _attr.IMAGE_START_TOKEN);

        _attr.IMAGE_ENCODER_INPUT_NCHW = -1;
        for (size_t i = 1; i < image_encoder.get_input(0).vShape.size(); i++) {
            if (image_encoder.get_input(0).vShape[i] == 3) {
                if (i == 1) {
                    _attr.IMAGE_ENCODER_INPUT_NCHW = 1;
                } else if (i == 3) {
                    _attr.IMAGE_ENCODER_INPUT_NCHW = 0;
                }
            }
        }
        if (_attr.IMAGE_ENCODER_INPUT_NCHW == -1) {
            ALOGE("image encoder input nchw or nhwc not found");
            return false;
        }

        if (_attr.IMAGE_ENCODER_INPUT_NCHW) {
            ALOGI("image encoder input nchw@float32");
            _attr.image_encoder_height = image_encoder.get_input(0).vShape[2];
            _attr.image_encoder_width  = image_encoder.get_input(0).vShape[3];
        } else {
            ALOGI("image encoder input nhwc@uint8");
            _attr.image_encoder_height = image_encoder.get_input(0).vShape[1];
            _attr.image_encoder_width  = image_encoder.get_input(0).vShape[2];
        }

        if (_attr.image_encoder_height != _attr.image_encoder_width) {
            ALOGE("image encoder height != width");
            return false;
        }
        int output_elem_size = 1;
        for (int i = 0; i < image_encoder.get_output(0).vShape.size(); i++) {
            output_elem_size *= image_encoder.get_output(0).vShape[i];
        }

        if (output_elem_size * 2 == image_encoder.get_output(0).nSize) {
            _attr.IMAGE_ENCODER_OUTPUT_BF16 = 1;
            ALOGI("image encoder output bf16");
        } else if (output_elem_size * 4 == image_encoder.get_output(0).nSize) {
            _attr.IMAGE_ENCODER_OUTPUT_BF16 = 0;
            ALOGI("image encoder output float32");
        } else {
            ALOGE("image encoder output not support");
            return false;
        }

        printf("\n");

        {
            ALOGI("image_encoder_height : %d, image_encoder_width: %d", _attr.image_encoder_height,
                  _attr.image_encoder_width);
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);
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
                ALOGI("grp: %d, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num =
                _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }
        nlohmann::json dynamic_config;

        dynamic_config["enable_temperature"] = _attr.enable_temperature;
        dynamic_config["temperature"]        = _attr.temperature;

        dynamic_config["enable_repetition_penalty"] = _attr.enable_repetition_penalty;
        dynamic_config["repetition_penalty"]        = _attr.repetition_penalty;
        dynamic_config["penalty_window"]            = _attr.penalty_window;

        dynamic_config["enable_top_p_sampling"] = _attr.enable_top_p_sampling;
        dynamic_config["top_p"]                 = _attr.top_p;

        dynamic_config["enable_top_k_sampling"] = _attr.enable_top_k_sampling;
        dynamic_config["top_k"]                 = _attr.top_k;

        if (!postprocess.load_config(attr.post_config_path)) {
            ALOGW("load postprocess config(%s) failed", attr.post_config_path.c_str());
        }

        if (!postprocess.load_config(dynamic_config)) {
            ALOGW("load postprocess config(%s) failed", dynamic_config.dump(4).c_str());
        }

        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    LLMPostprocess *getPostprocess()
    {
        return &postprocess;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++) {
            llama_layers[i].layer.release();
        }
        llama_post.release();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int SetSystemPrompt(std::string system_prompt, std::vector<int> &_token_ids)
    {
        tokenizer->Reset(system_prompt, _token_ids);
        _attr.system_prompt         = system_prompt;
        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
        return 0;
    }

    int GenerateKVCachePrefill(std::vector<int> &_token_ids, std::vector<std::vector<unsigned short>> &k_caches,
                               std::vector<std::vector<unsigned short>> &v_caches, int &precompute_len)
    {
        bfloat16 bf16       = -65536.f;
        int input_embed_num = _token_ids.size();
        precompute_len      = _token_ids.size();

        k_caches.resize(_attr.axmodel_num);
        v_caches.resize(_attr.axmodel_num);
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);

        int prefill_grpid = _attr.prefill_max_kv_cache_num_grp.size();

        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++) {
            if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i]) {
                prefill_grpid = i + 1;
                break;
            }
        }
        ALOGI("input token num : %d, prefill_split_num : %d prefill_grpid : %d", input_embed_num, prefill_split_num,
              prefill_grpid);

        for (size_t i = 0; i < _attr.axmodel_num; i++) {
            memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "K_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(prefill_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(prefill_grpid, "V_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(prefill_grpid, "V_cache").nSize);
        }

        if (input_embed_num == 0) {
            for (size_t i = 0; i < _attr.axmodel_num; i++) {
                k_caches[i].resize(precompute_len * _attr.kv_cache_size);
                v_caches[i].resize(precompute_len * _attr.kv_cache_size);
            }
            ALOGI("input token num is 0, skip");
            return 0;
        }

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];

        std::vector<unsigned short> test_embed;
        test_embed.resize(_token_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < _token_ids.size(); i++) {
            embed_selector.getByIndex(_token_ids[i], test_embed.data() + i * _attr.tokens_embed_size);
        }

        for (size_t p = 0; p < prefill_split_num; p++) {
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

                    for (int j = 0; j < p * _attr.prefill_token_num; j++) {
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
                    embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                    (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            } else {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                       _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++) {
                auto &layer = llama_layers[m];
                // set indices
                auto &input_indices             = layer.layer.get_input(prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                int idx = 0;
                for (unsigned int i = p * _attr.prefill_token_num; i < (p + 1) * _attr.prefill_token_num; i++) {
                    input_indices_ptr[idx] = i;
                    idx++;
                }

                // set mask
                auto &input_mask = layer.layer.get_input(prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(prefill_grpid, "V_cache_out");

                int kv_offset = (p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset, (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset, (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset, (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset, (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.prefill_token_num * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));
            }
        }

        for (size_t i = 0; i < _attr.axmodel_num; i++) {
            auto &layer = llama_layers[i];
            k_caches[i].resize(precompute_len * _attr.kv_cache_size);
            v_caches[i].resize(precompute_len * _attr.kv_cache_size);
            auto &input_k_cache = layer.layer.get_input(prefill_grpid, "K_cache");
            auto &input_v_cache = layer.layer.get_input(prefill_grpid, "V_cache");
            memcpy((void *)k_caches[i].data(), (void *)input_k_cache.pVirAddr,
                   precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
            memcpy((void *)v_caches[i].data(), (void *)input_v_cache.pVirAddr,
                   precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
        }

        return 0;
    }

    int GenerateKVCache(std::vector<int> &_token_ids)
    {
        for (size_t i = 0; i < _attr.axmodel_num; i++) {
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize);
        }

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        mask[_attr.kv_cache_num] = 0;
        std::vector<unsigned short> embed;

        int next_token = _token_ids[0];

        t_cqdm cqdm = create_cqdm(_token_ids.size(), 32);

        for (unsigned int indices = 0; indices < _token_ids.size(); indices++) {
            embed_selector.getByIndex(next_token, embed);

            for (int m = 0; m < _attr.axmodel_num; m++) {
                if (b_stop) {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache               = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                auto &input_v_cache               = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(decode_grpid, "input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(decode_grpid, "output");
                memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));
            }
            mask[indices] = 0;
            next_token    = _token_ids[indices + 1];
            update_cqdm(&cqdm, indices, "token", "");
            // ALOGI("");
        }
        return 0;
    }

    int GetKVCache(std::vector<std::vector<unsigned short>> &k_caches,
                   std::vector<std::vector<unsigned short>> &v_caches, int &precompute_len)
    {
        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        auto &input_mask = llama_layers[0].layer.get_input(decode_grpid, "mask");
        memcpy(mask.data(), (void *)input_mask.pVirAddr, input_mask.nSize);
        for (size_t i = 0; i < mask.size(); i++) {
            if (mask[i] == bf16.data) {
                precompute_len = i + 1;
                break;
            }
        }
        ALOGI("precompute_len:%d, remaining:%d", precompute_len,
              _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1] - precompute_len);
        k_caches.resize(_attr.axmodel_num);
        v_caches.resize(_attr.axmodel_num);
        for (size_t i = 0; i < _attr.axmodel_num; i++) {
            auto &layer = llama_layers[i];
            k_caches[i].resize(precompute_len * _attr.kv_cache_size);
            v_caches[i].resize(precompute_len * _attr.kv_cache_size);
            auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
            auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");
            memcpy((void *)k_caches[i].data(), (void *)input_k_cache.pVirAddr,
                   precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
            memcpy((void *)v_caches[i].data(), (void *)input_v_cache.pVirAddr,
                   precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
        }

        _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];

        return 0;
    }

    int SetKVCache(std::vector<std::vector<unsigned short>> &k_caches,
                   std::vector<std::vector<unsigned short>> &v_caches, int precompute_len, int input_num_token)
    {
        _attr.precompute_len = precompute_len;
        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++) {
            if (_attr.precompute_len + input_num_token <= _attr.prefill_max_kv_cache_num_grp[i]) {
                _attr.prefill_grpid = i + 1;
                break;
            }
        }
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];
        ALOGI("prefill_grpid:%d kv_cache_num:%d precompute_len:%d input_num_token:%d", _attr.prefill_grpid,
              kv_cache_num, precompute_len, input_num_token);

        _attr.prefill_max_token_num =
            ALIGN_DOWN(_attr.prefill_max_token_num - _attr.precompute_len, _attr.prefill_token_num);
        ALOGI("current prefill_max_token_num:%d", _attr.prefill_max_token_num);

        if (precompute_len == 0) {
            ALOGI("first run");
            return 0;
        }

        if (precompute_len + input_num_token > kv_cache_num) {
            ALOGE("precompute_len(%d) + input_num_token(%d) > _attr.prefill_max_kv_cache_num_grp[%d]", precompute_len,
                  input_num_token, _attr.prefill_grpid - 1);
            return -1;
        }

        if (input_num_token > _attr.prefill_max_token_num) {
            ALOGE("input_num_token(%d) > _attr.prefill_max_token_num(%d)", input_num_token,
                  _attr.prefill_max_token_num);
            return -1;
        }

        if (k_caches.size() != v_caches.size()) {
            ALOGE("k_caches.size(%d) != v_caches.size(%d)", k_caches.size(), v_caches.size());
            return -1;
        }

        if (k_caches.size() != _attr.axmodel_num) {
            ALOGE("k_caches.size(%d) != _attr.axmodel_num(%d)", k_caches.size(), _attr.axmodel_num);
            return -1;
        }

        // clear kv cache
        for (size_t i = 0; i < _attr.axmodel_num; i++) {
            memset((void *)llama_layers[i].layer.get_input(_attr.prefill_grpid, "K_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(_attr.prefill_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(_attr.prefill_grpid, "V_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(_attr.prefill_grpid, "V_cache").nSize);

            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "K_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(decode_grpid, "K_cache").nSize);
            memset((void *)llama_layers[i].layer.get_input(decode_grpid, "V_cache").pVirAddr, 0,
                   llama_layers[i].layer.get_input(decode_grpid, "V_cache").nSize);
        }

        for (unsigned int m = 0; m < _attr.axmodel_num; m++) {
            auto &layer = llama_layers[m];

            auto &k_cache = k_caches[m];
            auto &v_cache = v_caches[m];

            if (k_cache.size() != _attr.precompute_len * _attr.kv_cache_size) {
                ALOGE("k_cache.size(%d) != precompute_len(%d) * _attr.kv_cache_size(%d)", k_cache.size(),
                      _attr.precompute_len, _attr.kv_cache_size);
                return -1;
            }
            if (v_cache.size() < _attr.precompute_len * _attr.kv_cache_size) {
                ALOGE("v_cache.size(%d) < precompute_len(%d) * _attr.kv_cache_size(%d)", v_cache.size(),
                      _attr.precompute_len, _attr.kv_cache_size);
                return -1;
            }

            // set kv cache inputs
            {
                auto &input_k_cache               = layer.layer.get_input(_attr.prefill_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                auto &input_v_cache               = layer.layer.get_input(_attr.prefill_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                memcpy(input_k_cache_ptr, k_cache.data(),
                       _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                memcpy(input_v_cache_ptr, v_cache.data(),
                       _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
            }

            {
                auto &input_k_cache               = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                auto &input_v_cache               = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                memcpy(input_k_cache_ptr, k_cache.data(),
                       _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
                memcpy(input_v_cache_ptr, v_cache.data(),
                       _attr.precompute_len * _attr.kv_cache_size * sizeof(unsigned short));
            }
        }

        return 0;
    }

    bool save_kvcache(std::string target_path, std::string system_prompt, int precompute_len,
                      std::vector<std::vector<unsigned short>> &k_caches,
                      std::vector<std::vector<unsigned short>> &v_caches)
    {
        for (size_t i = 0; i < k_caches.size(); i++) {
            std::string k_cache_path = target_path + "/k_cache_" + std::to_string(i) + ".bin";
            std::string v_cache_path = target_path + "/v_cache_" + std::to_string(i) + ".bin";
            std::ofstream k_cache_file(k_cache_path);
            std::ofstream v_cache_file(v_cache_path);
            if (!k_cache_file.is_open() || !v_cache_file.is_open()) {
                ALOGE("save kvcache failed");
                return false;
            }
            k_cache_file.write((char *)k_caches[i].data(), k_caches[i].size() * sizeof(unsigned short));
            v_cache_file.write((char *)v_caches[i].data(), v_caches[i].size() * sizeof(unsigned short));
            k_cache_file.close();
            v_cache_file.close();
        }
        nlohmann::json j;
        j["system_prompt"]      = system_prompt;
        j["precompute_len"]     = precompute_len;
        std::string config_path = target_path + "/config.json";
        std::ofstream config_file(config_path);
        config_file << j.dump();
        config_file.close();
        return true;
    }

    bool load_kvcache(std::string target_path, int axmodel_num, std::vector<std::vector<unsigned short>> &k_caches,
                      std::vector<std::vector<unsigned short>> &v_caches, std::string &system_prompt,
                      int &precompute_len)
    {
        k_caches.resize(axmodel_num);
        v_caches.resize(axmodel_num);
        for (size_t i = 0; i < k_caches.size(); i++) {
            std::string k_cache_path = target_path + "/k_cache_" + std::to_string(i) + ".bin";
            std::string v_cache_path = target_path + "/v_cache_" + std::to_string(i) + ".bin";
            if (file_exist(k_cache_path) && file_exist(v_cache_path)) {
                std::vector<unsigned short> k_cache;
                std::vector<unsigned short> v_cache;
                std::ifstream k_cache_file(k_cache_path);
                std::ifstream v_cache_file(v_cache_path);

                k_cache_file.seekg(0, std::ios::end);
                k_cache.resize(k_cache_file.tellg() / sizeof(unsigned short));
                k_cache_file.seekg(0, std::ios::beg);

                v_cache_file.seekg(0, std::ios::end);
                v_cache.resize(v_cache_file.tellg() / sizeof(unsigned short));
                v_cache_file.seekg(0, std::ios::beg);

                k_cache_file.read((char *)k_cache.data(), k_cache.size() * sizeof(unsigned short));
                v_cache_file.read((char *)v_cache.data(), v_cache.size() * sizeof(unsigned short));

                k_cache_file.close();
                v_cache_file.close();
                k_caches[i] = k_cache;
                v_caches[i] = v_cache;
            } else {
                ALOGE("k_cache %s or v_cache %s not exist", k_cache_path.c_str(), v_cache_path.c_str());
                return false;
            }
        }

        std::string config_path = target_path + "/config.json";
        if (file_exist(config_path)) {
            std::ifstream config_file(config_path);
            nlohmann::json j;
            config_file >> j;
            system_prompt  = j["system_prompt"].get<std::string>();
            precompute_len = j["precompute_len"].get<int>();
            config_file.close();
        } else {
            ALOGE("config %s not exist", config_path.c_str());
            return false;
        }
        return true;
    }

    int Encode(cv::Mat src, std::vector<unsigned short> &out_embed)
    {
        timer t;
        t.start();
        if (_attr.IMAGE_ENCODER_INPUT_NCHW) {
            std::vector<float> mean  = {0.485, 0.456, 0.406};
            std::vector<float> scale = {0.229, 0.224, 0.225};

            cv::Mat dst;
            cv::resize(src, dst, cv::Size(_attr.image_encoder_width, _attr.image_encoder_height));
            cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

            float *input_data = (float *)image_encoder.get_input(0).pVirAddr;

            unsigned char *img_data = dst.data;
            int letterbox_rows      = dst.rows;
            int letterbox_cols      = dst.cols;

            for (int h = 0; h < letterbox_rows; h++) {
                for (int w = 0; w < letterbox_cols; w++) {
                    for (int c = 0; c < 3; c++) {
                        int in_index          = h * letterbox_cols * 3 + w * 3 + c;
                        int out_index         = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                        input_data[out_index] = (float(img_data[in_index]) / 255.0 - mean[c]) / scale[c];
                    }
                }
            }
            image_encoder.inference();
        } else {
            cv::Mat dst;
            cv::resize(src, dst, cv::Size(_attr.image_encoder_width, _attr.image_encoder_height));
            cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
            void *data = image_encoder.get_input(0).pVirAddr;
            memcpy(data, dst.data, dst.rows * dst.cols * 3);
            image_encoder.inference();
        }

        int size = 1;
        for (size_t i = 0; i < image_encoder.get_output(0).vShape.size(); i++) {
            size *= image_encoder.get_output(0).vShape[i];
        }

        out_embed.resize(size);

        if (_attr.IMAGE_ENCODER_OUTPUT_BF16)
            memcpy(out_embed.data(), image_encoder.get_output(0).pVirAddr, image_encoder.get_output(0).nSize);
        else {
            float *out_data = (float *)image_encoder.get_output(0).pVirAddr;
            for (size_t i = 0; i < size; i++) {
                out_embed[i] = bfloat16(out_data[i]).data;
            }
        }

        ALOGI("image encode time : %0.2f ms, size : %ld", t.cost(), out_embed.size());
        return 0;
    }

    int Encode(std::vector<cv::Mat> srcs, std::vector<std::vector<unsigned short>> &out_embeds)
    {
        out_embeds.resize(srcs.size());
        for (size_t i = 0; i < srcs.size(); i++) {
            auto ret = Encode(srcs[i], out_embeds[i]);
            if (ret != 0) {
                ALOGE("Encode image failed");
                return -1;
            }
        }

        return 0;
    }

    int Encode(std::vector<std::vector<unsigned short>> &imgs_embed, std::vector<unsigned short> &out_embed,
               std::string prompt, std::vector<int> &tokens_ids, std::vector<int> &tokens_diff)
    {
        ImageInfo img_info;
        img_info.img_prompt        = true;
        img_info.num_img           = imgs_embed.size();
        img_info.imgsz             = _attr.image_encoder_width;
        std::vector<int> input_ids = tokenizer->Encode_ctx(prompt, img_info, tokens_ids, tokens_diff);

        std::vector<int> img_start_index;
        for (size_t i = 0; i < input_ids.size(); i++) {
            if (input_ids[i] == _attr.IMAGE_START_TOKEN) {
                img_start_index.push_back(i);
            }
        }

        if (img_start_index.size() != imgs_embed.size()) {
            ALOGE("img_start_index.size() != imgs_embed.size(), img_start_index.size() : %ld, imgs_embed.size() : %ld",
                  img_start_index.size(), imgs_embed.size());

            printf("input_ids : ");
            for (size_t i = 0; i < input_ids.size(); i++) {
                printf("%d ", input_ids[i]);
            }
            printf("\n");

            return -1;
        }

        if (input_ids.size() > _attr.prefill_max_token_num) {
            ALOGE("input_ids(%ld) > prefill_max_token_num(%d)", input_ids.size(), _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++) {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }
        for (size_t i = 0; i < imgs_embed.size(); i++) {
            int offset      = img_start_index[i] + 1;
            auto &img_embed = imgs_embed[i];

            int img_context_count = 0;
            for (size_t j = offset; j < input_ids.size(); j++) {
                if (input_ids[j] == _attr.IMAGE_CONTEXT_TOKEN) {
                    img_context_count++;
                } else {
                    break;
                }
            }

            if (img_context_count != img_embed.size() / _attr.tokens_embed_size) {
                ALOGE("img_context_count(%d) != img_embed.size() / tokens_embed_size(%ld)", img_context_count,
                      img_embed.size() / _attr.tokens_embed_size);
                return -1;
            }

            memcpy(out_embed.data() + offset * _attr.tokens_embed_size, img_embed.data(),
                   img_embed.size() * sizeof(unsigned short));
            ALOGI("idx:%ld offset : %d out_embed.size() : %ld", i, offset, out_embed.size());
        }

        return 0;
    }

    int Encode(std::vector<unsigned short> &img_embed, std::vector<unsigned short> &out_embed, std::string prompt,
               std::vector<int> &tokens_ids, std::vector<int> &tokens_diff)
    {
        // std::vector<std::vector<unsigned short>> imgs_embed = {img_embed};
        imgs_embed_.push_back(img_embed);
        return Encode(imgs_embed_, out_embed, prompt, tokens_ids, tokens_diff);
    }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt, std::string last_reply,
               std::vector<int> &tokens_ids, std::vector<int> &tokens_diff)
    {
        ImageInfo img_info;
        img_info.img_prompt = false;
        if (!tokenizer->Encode(prompt, last_reply, tokens_ids, tokens_diff, img_info)) {
            ALOGE("encode failed");
            return -1;
        }

        out_embed.resize(tokens_diff.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < tokens_diff.size(); i++) {
            embed_selector.getByIndex(tokens_diff[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        return 0;
    }

    void ClearImgsEmbed()
    {
        imgs_embed_.clear();
    }

    std::string Run(std::vector<unsigned short> &test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_grpid - 1];

        std::vector<int> cached_token;
        std::vector<int> token_ids;

        int input_embed_num   = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < _attr.precompute_len + input_embed_num; i++) {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        for (size_t p = 0; p < prefill_split_num; p++) {
            if (b_stop) {
                break;
            }

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
                    embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                    (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            } else {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size,
                       _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++) {
                if (b_stop) {
                    break;
                }

                auto &layer = llama_layers[m];

                // set indices
                auto &input_indices             = layer.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                int idx = 0;
                for (unsigned int i = _attr.precompute_len + p * _attr.prefill_token_num;
                     i < _attr.precompute_len + (p + 1) * _attr.prefill_token_num; i++) {
                    input_indices_ptr[idx] = i;
                    idx++;
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

                auto &input_prefill_k_cache = layer.layer.get_input(_attr.prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer.get_input(_attr.prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (_attr.precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset, (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset, (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset, (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset, (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));
            }
            if (p == (prefill_split_num - 1)) {
                memcpy(embed.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }

        int next_token = -1;
        t_cqdm cqdm    = create_cqdm(_attr.max_token_len, 32);

        {
            // post process
            auto &input = llama_post.get_input("input");
            memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();
            int max_index;

            auto &output_post = llama_post.get_output("output");
            // AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
            unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
            float max_val            = -MAXFLOAT;
            // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
            max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

            next_token = max_index;

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;
        for (unsigned int indices = _attr.precompute_len + input_embed_num; indices < _attr.max_token_len; indices++) {
            if (b_stop) {
                break;
            }

            embed_selector.getByIndex(next_token, embed);

            for (int m = 0; m < _attr.axmodel_num; m++) {
                if (b_stop) {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache               = layer.layer.get_input(decode_grpid, "K_cache");
                unsigned short *input_k_cache_ptr = (unsigned short *)input_k_cache.pVirAddr;
                auto &input_v_cache               = layer.layer.get_input(decode_grpid, "V_cache");
                unsigned short *input_v_cache_ptr = (unsigned short *)input_v_cache.pVirAddr;

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy(input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                auto &input_input = layer.layer.get_input(decode_grpid, "input");
                memcpy(input_input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr,
                       sizeof(unsigned short) * _attr.kv_cache_size);

                auto &output = layer.layer.get_output(decode_grpid, "output");
                memcpy(embed.data(), output.pVirAddr, embed.size() * sizeof(unsigned short));
            }
            mask[indices] = 0;
            {
                // post process
                auto &input = llama_post.get_input("input");
                memcpy(input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
                llama_post.inference();
                int max_index;

                auto &output_post = llama_post.get_output("output");
                // AX_SYS_MinvalidateCache(output_post.phyAddr, output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val            = -MAXFLOAT;
                // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
                max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

                next_token = max_index;

                if (tokenizer->isEnd(max_index)) {
                    if (cached_token.size() && _attr.runing_callback) {
                        float t_cost_ms     = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out        = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec,
                                              _attr.reserve);
                        cached_token.clear();
                    }
                    b_hit_eos = true;
                    break;
                }
                token_ids.push_back(max_index);

                if (_attr.runing_callback) {
                    cached_token.push_back(max_index);
                    if (cached_token.size() >= 3) {
                        float t_cost_ms     = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out        = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec,
                                              _attr.reserve);
                        cached_token.clear();
                    }
                }
            }

            if (_attr.runing_callback == nullptr) update_cqdm(&cqdm, indices, "token", "");
            if (b_hit_eos) {
                break;
            }
        }
        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_cost_ms / 1000));

        final_out = tokenizer->Decode(token_ids);

        return final_out;
    }
};

class LLM_Qwen {
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

    LLMAttrType _attr;

    struct LLMLayer {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;
    ax_runner_ax650 image_encoder;
    int deepstack_features_num = 3;
    // int prefill_grpid = 1;
    int decode_grpid = 0;

    bool b_stop = false;

    LLMPostprocess postprocess;
    static int post_process(LLMPostprocess &postprocess, unsigned short *p, int n, std::vector<int> &history,
                            float *val = 0)
    {
        std::vector<float> logits(n);
        for (int i = 0; i < n; i++) {
            unsigned int proc = p[i] << 16;
            logits[i]         = *reinterpret_cast<float *>(&proc);
        }

        return postprocess.apply(logits, history);
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        int remain_cmm = get_remaining_cmm_size();
        ALOGI("Total CMM:%d MB", remain_cmm);

        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer   = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.url_tokenizer_model, attr.b_bos, attr.b_eos)) {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.url_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size,
                                 attr.b_use_mmap_load_embed)) {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num,
                  attr.tokens_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");

        llama_layers.resize(attr.axmodel_num);

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
        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        ret = image_encoder.init(attr.filename_image_encoder_axmodel.c_str(), false);
        if (ret != 0) {
            ALOGE("init image_encoder axmodel(%s) failed", attr.filename_image_encoder_axmodel.c_str());
            return false;
        }

        remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init vpm axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 3, "count", axmodel_path);

        _attr.IMAGE_CONTEXT_TOKEN = tokenizer->GetImgContextID();
        _attr.IMAGE_START_TOKEN   = tokenizer->GetImgStartID();

        ALOGI("IMAGE_CONTEXT_TOKEN: %d, IMAGE_START_TOKEN: %d", _attr.IMAGE_CONTEXT_TOKEN, _attr.IMAGE_START_TOKEN);

        _attr.IMAGE_ENCODER_INPUT_NCHW = -1;
        for (size_t i = 1; i < image_encoder.get_input(0).vShape.size(); i++) {
            if (image_encoder.get_input(0).vShape[i] == 3) {
                if (i == 1) {
                    _attr.IMAGE_ENCODER_INPUT_NCHW = 1;
                } else if (i == 3) {
                    _attr.IMAGE_ENCODER_INPUT_NCHW = 0;
                }
            }
        }
        if (_attr.IMAGE_ENCODER_INPUT_NCHW == -1) {
            ALOGE("image encoder input nchw or nhwc not found");
            return false;
        }

        if (_attr.IMAGE_ENCODER_INPUT_NCHW == 1) {
            ALOGE("Qwen2.5_VL Image Encoder just support NHWC");
            return false;
        }

        int output_elem_size = 1;
        for (int i = 0; i < image_encoder.get_output(0).vShape.size(); i++) {
            output_elem_size *= image_encoder.get_output(0).vShape[i];
        }

        if (output_elem_size * 2 == image_encoder.get_output(0).nSize) {
            _attr.IMAGE_ENCODER_OUTPUT_BF16 = 1;
            ALOGI("image encoder output bf16");
        } else if (output_elem_size * 4 == image_encoder.get_output(0).nSize) {
            _attr.IMAGE_ENCODER_OUTPUT_BF16 = 0;
            ALOGI("image encoder output float32");
        } else {
            ALOGE("image encoder output not support");
            return false;
        }

        if (attr.b_dynamic_load_axmodel_layer) {
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

        nlohmann::json dynamic_config;

        dynamic_config["enable_temperature"] = _attr.enable_temperature;
        dynamic_config["temperature"]        = _attr.temperature;

        dynamic_config["enable_repetition_penalty"] = _attr.enable_repetition_penalty;
        dynamic_config["repetition_penalty"]        = _attr.repetition_penalty;
        dynamic_config["penalty_window"]            = _attr.penalty_window;

        dynamic_config["enable_top_p_sampling"] = _attr.enable_top_p_sampling;
        dynamic_config["top_p"]                 = _attr.top_p;

        dynamic_config["enable_top_k_sampling"] = _attr.enable_top_k_sampling;
        dynamic_config["top_k"]                 = _attr.top_k;

        if (!postprocess.load_config(attr.post_config_path)) {
            ALOGW("load postprocess config(%s) failed", attr.post_config_path.c_str());
        }

        if (!postprocess.load_config(dynamic_config)) {
            ALOGW("load postprocess config(%s) failed", dynamic_config.dump(4).c_str());
        }

        ALOGI("LLM init ok");
        remain_cmm = get_remaining_cmm_size();
        ALOGI("Left CMM:%d MB", remain_cmm);
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    LLMPostprocess *getPostprocess()
    {
        return &postprocess;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++) {
            llama_layers[i].layer.release();
        }
        llama_post.release();
        image_encoder.release();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int EncodeImage(std::vector<cv::Mat> &src, bool b_video, Config &cfg,
                    std::vector<std::vector<unsigned short>> &out_embed,
                    std::vector<std::vector<float>> &deepstack_features)
    {
        int temporal_patch_size = cfg.vision_config.temporal_patch_size;
        int merge_size          = cfg.vision_config.spatial_merge_size;
        int patch_size          = cfg.vision_config.patch_size;
        int ret;
        timer t;
        t.start();

        unsigned int grid_h = cfg.vision_config.height / cfg.vision_config.patch_size;
        unsigned int grid_w = cfg.vision_config.width / cfg.vision_config.patch_size;
        std::vector<std::vector<unsigned char>> pixel_values;
        int w = cfg.vision_config.width, h = cfg.vision_config.height;
        int channel = src[0].channels();
        int hwc     = grid_h * grid_w * temporal_patch_size * patch_size * patch_size * channel;

        if (!b_video) {
            for (int i = 0; i < src.size(); i++) {
                std::vector<std::vector<unsigned char>> img_values;
                std::vector<cv::Mat> si{src[i]};
                Qwen2VideoProcessor(si, img_values, h, w, temporal_patch_size, merge_size, patch_size);
                pixel_values.push_back(img_values[0]);
            }
            for (size_t i = 0; i < pixel_values.size(); i++) {
                cfg.image_grid_thw.push_back({1, static_cast<int>(grid_h), static_cast<int>(grid_w)});
            }
        } else {
            Qwen2VideoProcessor(src, pixel_values, h, w, temporal_patch_size, merge_size, patch_size);
            cfg.video_grid_thw = {
                {static_cast<int>(pixel_values.size()), static_cast<int>(grid_h), static_cast<int>(grid_w)}};
        }

        ALOGI("pixel_values size %d", pixel_values.size());
        ALOGI("grid_h %d grid_w %d", grid_h, grid_w);
        int cnt = 0;
        if (out_embed.empty()) {
            out_embed.resize(pixel_values.size());
        }

        deepstack_features.clear();
        for (int i = 0; i < pixel_values.size(); i++) {
            void *data = image_encoder.get_input(0).pVirAddr;
            memcpy(data, pixel_values[i].data(), hwc);
            image_encoder.inference();

            size_t size = image_encoder.get_output(0).nSize / sizeof(float);
            if (out_embed[i].empty()) {
                out_embed[i].resize(size);
            }

            float *output_data = (float *)image_encoder.get_output(0).pVirAddr;
            for (size_t j = 0; j < size; j++) {
                out_embed[i][j] = bfloat16(output_data[j]).data;
            }

            for (int j = 0; j < deepstack_features_num; j++) {
                size_t size = image_encoder.get_output(j + 1).nSize / sizeof(float);
                std::vector<float> feature(size);

                float *output_data = (float *)image_encoder.get_output(j + 1).pVirAddr;
                for (size_t k = 0; k < size; k++) {
                    feature[k] = output_data[k];
                }

                // 将不同image 的 deepstack feature 拼接到一起
                if (i == 0) {
                    deepstack_features.push_back(feature);
                } else {
                    deepstack_features[j].insert(deepstack_features[j].end(), feature.begin(), feature.end());
                }
            }
        }

        ALOGI("image encode time : %f ms, size : %d", t.cost(), out_embed.size());
        return 0;
    }

    int GetPositionIds(std::vector<int> &input_ids, std::vector<std::vector<int>> &position_ids, Config &cfg)
    {
        position_ids = get_rope_index(cfg, input_ids, cfg.image_grid_thw, cfg.video_grid_thw);
        return 0;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::vector<std::vector<int>> &position_ids, Config &cfg,
               std::string prompt = "What is in the image?")
    {
        ImageInfo img_info;
        img_info.img_prompt        = false;
        std::vector<int> input_ids = tokenizer->Encode(prompt, img_info);
        if (input_ids.size() > _attr.prefill_max_token_num) {
            ALOGE("input_ids(%d) > prefill_max_token_num(%d)", input_ids.size(), _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++) {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        cfg.image_grid_thw.clear();
        cfg.video_grid_thw.clear();
        GetPositionIds(input_ids, position_ids, cfg);
        return 0;
    }

    int Encode(std::vector<std::vector<unsigned short>> &img_embed, bool b_video,
               std::vector<unsigned short> &out_embed, std::vector<std::vector<int>> &position_ids,
               std::vector<int> &visual_pos_mask, Config &cfg, std::string prompt = "What is in the image?")
    {
        ImageInfo img_info;
        img_info.img_prompt        = true;
        img_info.type              = b_video ? ImgType::Video : ImgType::Image;
        img_info.img_token_num     = img_embed[0].size() / _attr.tokens_embed_size;
        img_info.num_img           = img_embed.size();
        std::vector<int> input_ids = tokenizer->Encode(prompt, img_info);
        ALOGI("input_ids size:%d", input_ids.size());
        std::vector<int> offsets;
        int vision_start_token_id = cfg.vision_start_token_id;
        for (size_t i = 0; i < input_ids.size() - 1; i++) {
            if (input_ids[i] == vision_start_token_id) {
                int offset = i + 1;
                ALOGI("offset %d", offset);
                offsets.push_back(offset);
            }
        }

        visual_pos_mask.resize(input_ids.size());
        for (size_t i = 0; i < input_ids.size(); i++) {
            if (input_ids[i] == cfg.image_token_id || input_ids[i] == cfg.video_token_id) {
                visual_pos_mask[i] = 1;
            } else {
                visual_pos_mask[i] = 0;
            }
        }

        if (input_ids.size() > _attr.prefill_max_token_num) {
            ALOGE("input_ids(%ld) > prefill_max_token_num(%d)", input_ids.size(), _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(input_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < input_ids.size(); i++) {
            embed_selector.getByIndex(input_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }
        ALOGI("img_embed.size:%d, %d", img_embed.size(), img_embed[0].size());

        if (offsets.size() == 1 && img_embed.size() > 1) {
            for (int i = 1; i < img_embed.size(); i++) {
                offsets.push_back(offsets[i - 1] + img_embed[i - 1].size() / _attr.tokens_embed_size);
                ALOGI("offset:%d", offsets[i - 1] + img_embed[i - 1].size() / _attr.tokens_embed_size);
            }
        }

        for (int i = 0; i < img_embed.size(); i++) {
            memcpy(out_embed.data() + offsets[i] * _attr.tokens_embed_size, img_embed[i].data(),
                   img_embed[i].size() * sizeof(unsigned short));
        }

        ALOGI("out_embed size:%d", out_embed.size());
        ALOGI("input_ids size %d", input_ids.size());
        GetPositionIds(input_ids, position_ids, cfg);
        ALOGI("position_ids size:%d", position_ids[0].size());
        return 0;
    }

    std::string Run(std::vector<unsigned short> &test_embed, std::vector<std::vector<int>> &position_ids,
                    std::vector<std::vector<float>> &deepstack_features, std::vector<int> &visual_pos_mask)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);

        std::vector<int> cached_token;
        std::vector<int> token_ids;

        int input_embed_num   = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        if (input_embed_num > _attr.prefill_max_token_num) {
            ALOGE("input token num(%d) > prefill_max_token_num(%d)", input_embed_num, _attr.prefill_max_token_num);
            return "";
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
            int start, offset;

            start = p * _attr.prefill_token_num;
            if (p == (prefill_split_num - 1)) {
                offset = (input_embed_num - p * _attr.prefill_token_num);
                memcpy(embed_tmp.data(), test_embed.data() + start * _attr.tokens_embed_size,
                       offset * _attr.tokens_embed_size * sizeof(unsigned short));
            } else {
                offset = _attr.prefill_token_num;
                memcpy(embed_tmp.data(), test_embed.data() + start * _attr.tokens_embed_size,
                       offset * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            int start_deepstack_feat = 0, offset_deepstack_feat = 0;
            if (!visual_pos_mask.empty()) {
                for (int j = 0; j < start; j++) {
                    start_deepstack_feat += visual_pos_mask[j];
                }
                for (int j = start; j < start + offset; j++) {
                    offset_deepstack_feat += visual_pos_mask[j];
                }
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

                if (!visual_pos_mask.empty() && m < deepstack_features.size()) {
                    int k = 0;
                    for (int j = start, k = start_deepstack_feat; j < start + offset; j++) {
                        if (visual_pos_mask[j] == 0) {
                            continue;
                        }
                        for (int di = 0; di < _attr.tokens_embed_size; di++) {
                            // bfloat16 to float32
                            unsigned int tmp_bf16_1 = embed_tmp[(j - start) * _attr.tokens_embed_size + di] << 16;
                            float tmp_fp32_1        = *reinterpret_cast<float *>(&tmp_bf16_1);

                            float tmp_fp32_2 = deepstack_features[m][k * _attr.tokens_embed_size + di];
                            // float32 to bfloat16
                            embed_tmp[(j - start) * _attr.tokens_embed_size + di] =
                                bfloat16(tmp_fp32_1 + tmp_fp32_2).data;
                        }
                        k++;
                    }
                }

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

        {
            auto &input = llama_post.get_input(0);
            memcpy((void *)input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();

            int max_index;

            auto &output_post = llama_post.get_output(0);
            memcpy(output_post.pVirAddr, (void *)output_post.pVirAddr, output_post.nSize);
            unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
            float max_val            = -MAXFLOAT;
            max_index                = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, &max_val);

            next_token = max_index;

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;

        for (unsigned int indices = max_pos_id + 1; indices < _attr.max_token_len; indices++) {
            if (b_stop) {
                break;
            }

            embed_selector.getByIndex(next_token, embed);

            memcpy((void *)llama_layers[0].layer.get_input(decode_grpid, "input").pVirAddr, embed.data(),
                   llama_layers[0].layer.get_input(decode_grpid, "input").nSize);

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

                auto &output_post = llama_post.get_output(0);
                memcpy(output_post.pVirAddr, (void *)output_post.pVirAddr, output_post.nSize);
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val            = -MAXFLOAT;
                auto max_index = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);

                next_token = max_index;

                if (tokenizer->isEnd(max_index)) {
                    if (cached_token.size() && _attr.runing_callback) {
                        float t_cost_ms     = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out        = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec,
                                              _attr.reserve);
                        cached_token.clear();
                    }
                    b_hit_eos = true;
                    break;
                }
                token_ids.push_back(max_index);

                if (_attr.runing_callback) {
                    cached_token.push_back(max_index);
                    if (cached_token.size() >= 3) {
                        float t_cost_ms     = t_cost.cost();
                        float token_per_sec = token_ids.size() / (t_cost_ms / 1000);
                        auto tmp_out        = tokenizer->Decode(cached_token);
                        _attr.runing_callback(cached_token.data(), cached_token.size(), tmp_out.c_str(), token_per_sec,
                                              _attr.reserve);
                        cached_token.clear();
                    }
                }
            }

            if (_attr.runing_callback == nullptr) update_cqdm(&cqdm, indices, "token", "");
            if (b_hit_eos) {
                break;
            }
        }
        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_cost_ms / 1000));

        final_out = tokenizer->Decode(token_ids);

        for (size_t i = 0; i < _attr.axmodel_num; i++) {
            for (size_t j = 0; j < llama_layers[i].layer.get_num_input_groups(); j++) {
                memset((void *)llama_layers[i].layer.get_input(j, "K_cache").pVirAddr, 0,
                       llama_layers[i].layer.get_input(j, "K_cache").nSize);
                memset((void *)llama_layers[i].layer.get_input(j, "V_cache").pVirAddr, 0,
                       llama_layers[i].layer.get_input(j, "V_cache").nSize);
            }
        }

        return final_out;
    }
};