#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <memory>

#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "utils/utils.hpp"
#include "utils/slice_3d.h"
#include "utils/concat_3d.h"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "axcl_manager.h"
#include "BaseRunner.hpp"

struct Token2WavAttr {
    int devid                = 0;
    int flow_embed_num       = 6561;
    int flow_embed_size      = 512;
    int token_mel_ratio      = 2;
    int token_hop_len        = 25;
    int max_infer_chunk_num  = 3;
    int mel_cache_len        = 8;
    int source_cache_len     = mel_cache_len * 480;
    int pre_lookahead_len    = 3;
    float inference_cfg_rate = 0.7;

    std::string flow_input_embedding    = "flow.input_embedding.float16.bin";
    std::string rand_noise              = "rand_noise_1_80_300.txt";
    std::string speech_window           = "speech_window_2x8x480.txt";
    std::string flow_encoder_28         = "flow_encoder_28.axmodel";
    std::string flow_encoder_53         = "flow_encoder_53.axmodel";
    std::string flow_encoder_78         = "flow_encoder_78.axmodel";
    std::string flow_encoder_50_final   = "flow_encoder_50_final.axmodel";
    std::string flow_estimator_200      = "flow_estimator_200.axmodel";
    std::string flow_estimator_250      = "flow_estimator_250.axmodel";
    std::string flow_estimator_300      = "flow_estimator_300.axmodel";
    std::string hift_p2_50_first        = "hift_p2_50_first.axmodel";
    std::string hift_p2_58              = "hift_p2_58.axmodel";
    std::string hift_p1_50_first        = "hift_p1_50_first.onnx";
    std::string hift_p1_58              = "hift_p1_58.onnx";
    std::string prompt_dir              = "prompt_data/";
    std::string prompt_text             = "prompt_text.txt";
    std::string llm_prompt_speech_token = "llm_prompt_speech_token.txt";
    std::string prompt_speech_feat      = "prompt_speech_feat.txt";
    std::string flow_embedding          = "flow_embedding.txt";
    int n_timesteps                     = 10;
};

class Token2Wav {
public:
    Token2WavAttr _attr;

private:
    ax_runner_ax650 flow_encoder_28;
    ax_runner_ax650 flow_encoder_53;
    ax_runner_ax650 flow_encoder_78;
    ax_runner_ax650 flow_encoder_50_final;
    ax_runner_ax650 flow_estimator_200;
    ax_runner_ax650 flow_estimator_250;
    ax_runner_ax650 flow_estimator_300;
    ax_runner_ax650 hift_p2_50_first;
    ax_runner_ax650 hift_p2_58;
    std::shared_ptr<BaseRunner> hift_p1_50_first;
    std::shared_ptr<BaseRunner> hift_p1_58;
    std::vector<float> rand_noise;
    std::vector<float> t_span;
    LLaMaEmbedSelector flow_embed_selector;
    std::unordered_map<std::string, std::vector<float>> hift_cache_dict;
    std::vector<float> speech_window;

    int init_noise(const std::string &path)
    {
        return readtxt(path, rand_noise);
    }
    int init_speech_window(const std::string &path)
    {
        return readtxt(path, speech_window);
    }
    int init_tspan(int n_timesteps)
    {
        if (n_timesteps < 4) return -1;
        t_span = linspace(0.0, 1.0, n_timesteps + 1);
        return 0;
    }

public:
    bool Init(const Token2WavAttr &attr)
    {
        _attr = attr;

        if (init_tspan(_attr.n_timesteps) != 0) {
            ALOGE("init_tspan failed, n_timesteps:%d", _attr.n_timesteps);
            return false;
        }
        if (init_noise(_attr.rand_noise) != 0) {
            ALOGE("init rand noise failed");
            return false;
        }
        if (init_speech_window(_attr.speech_window) != 0) {
            ALOGE("init speech_window failed");
            return false;
        }
        if (!flow_embed_selector.Init(_attr.flow_input_embedding.c_str(), _attr.flow_embed_num, _attr.flow_embed_size,
                                      false)) {
            ALOGE("flow_embed_selector init failed");
            return false;
        }
        if (flow_encoder_28.init(_attr.flow_encoder_28.c_str(), _attr.devid) != 0) return false;
        if (flow_encoder_53.init(_attr.flow_encoder_53.c_str(), _attr.devid) != 0) return false;
        if (flow_encoder_78.init(_attr.flow_encoder_78.c_str(), _attr.devid) != 0) return false;
        if (flow_encoder_50_final.init(_attr.flow_encoder_50_final.c_str(), _attr.devid) != 0) return false;
        if (flow_estimator_200.init(_attr.flow_estimator_200.c_str(), _attr.devid) != 0) return false;
        if (flow_estimator_250.init(_attr.flow_estimator_250.c_str(), _attr.devid) != 0) return false;
        if (flow_estimator_300.init(_attr.flow_estimator_300.c_str(), _attr.devid) != 0) return false;
        if (hift_p2_50_first.init(_attr.hift_p2_50_first.c_str(), _attr.devid) != 0) return false;
        if (hift_p2_58.init(_attr.hift_p2_58.c_str(), _attr.devid) != 0) return false;

        hift_p1_50_first = CreateRunner(RT_OnnxRunner);
        if (!hift_p1_50_first) return false;
        BaseConfig cfg50;
        cfg50.nthread    = 8;
        cfg50.onnx_model = _attr.hift_p1_50_first;
        hift_p1_50_first->load(cfg50);

        hift_p1_58 = CreateRunner(RT_OnnxRunner);
        if (!hift_p1_58) return false;
        BaseConfig cfg58;
        cfg58.nthread    = 8;
        cfg58.onnx_model = _attr.hift_p1_58;
        hift_p1_58->load(cfg58);

        return true;
    }

    void Deinit()
    {
        flow_encoder_28.release();
        flow_encoder_53.release();
        flow_encoder_78.release();
        flow_encoder_50_final.release();
        flow_estimator_200.release();
        flow_estimator_250.release();
        flow_estimator_300.release();
        hift_p2_50_first.release();
        hift_p2_58.release();
        flow_embed_selector.Deinit();
    }

    void SetTimesteps(int n_timesteps)
    {
        init_tspan(n_timesteps);
    }

    int SpeechToken2Embeds(std::vector<int> &token_ids, std::vector<float> &token_embeds)
    {
        if (token_embeds.empty() || token_embeds.size() != token_ids.size() * _attr.flow_embed_size) {
            token_embeds.resize(token_ids.size() * _attr.flow_embed_size);
        }
        std::vector<unsigned short> speech_embeds_one(_attr.flow_embed_size);
        for (size_t i = 0; i < token_ids.size(); i++) {
            flow_embed_selector.getByIndex(token_ids[i], speech_embeds_one.data());
            for (int j = 0; j < _attr.flow_embed_size; j++) {
                unsigned int proc                           = speech_embeds_one[j] << 16;
                token_embeds[i * _attr.flow_embed_size + j] = *reinterpret_cast<float *>(&proc);
            }
        }
        return token_embeds.size();
    }

    int infer_flow_encoder(std::vector<float> &token_embeds, std::vector<float> &prompt_feat,
                           std::vector<float> &spk_embeds, int token_len, bool finalize, std::vector<float> &mu,
                           std::vector<float> &spks, std::vector<float> &cond)
    {
        ax_runner_ax650 *model;
        if (!finalize) {
            if (token_len == 28) {
                model = &flow_encoder_28;
            } else if (token_len == 53) {
                model = &flow_encoder_53;
            } else if (token_len == 78) {
                model = &flow_encoder_78;
            } else {
                return -1;
            }
        } else if (token_len == 50) {
            model = &flow_encoder_50_final;
        } else {
            return -1;
        }

        void *p = (void *)model->get_input("token_embedding").phyAddr;
        axcl_Memcpy(p, token_embeds.data(), token_embeds.size() * sizeof(float),
                    axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);
        p = (void *)model->get_input("prompt_feat").phyAddr;
        axcl_Memcpy(p, prompt_feat.data(), prompt_feat.size() * sizeof(float),
                    axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);
        p = (void *)model->get_input("embedding").phyAddr;
        axcl_Memcpy(p, spk_embeds.data(), spk_embeds.size() * sizeof(float),
                    axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);

        model->inference();

        auto &output_mu = model->get_output("mu");
        if (mu.empty()) {
            mu.resize(output_mu.nSize / sizeof(float));
        }
        // axcl_Memcpy(mu.data(), (void *)output_mu.phyAddr, output_mu.nSize,
        // axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, devid);
        axcl_Memcpy((void *)output_mu.pVirAddr, (void *)output_mu.phyAddr, output_mu.nSize,
                    axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, _attr.devid);
        memcpy(mu.data(), (void *)output_mu.pVirAddr, output_mu.nSize);

        auto &output_spks = model->get_output("spks");
        if (spks.empty()) {
            spks.resize(output_spks.nSize / sizeof(float));
        }
        // axcl_Memcpy(spks.data(), (void *)output_spks.phyAddr, output_spks.nSize,
        // axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, devid);
        axcl_Memcpy((void *)output_spks.pVirAddr, (void *)output_spks.phyAddr, output_spks.nSize,
                    axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, _attr.devid);
        memcpy(spks.data(), (void *)output_spks.pVirAddr, output_spks.nSize);

        auto &output_cond = model->get_output("cond");
        if (cond.empty()) {
            cond.resize(output_cond.nSize / sizeof(float));
        }
        // axcl_Memcpy(cond.data(), (void *)output_cond.phyAddr, output_cond.nSize,
        // axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, devid);
        axcl_Memcpy((void *)output_cond.pVirAddr, (void *)output_cond.phyAddr, output_cond.nSize,
                    axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, _attr.devid);
        mempcpy(cond.data(), (void *)output_cond.pVirAddr, output_cond.nSize);

        return 0;
    }

    int infer_flow_estimator(std::vector<float> &x, std::vector<float> &mask, std::vector<float> &t,
                             std::vector<float> &mu, std::vector<float> &spks, std::vector<float> &cond,
                             std::vector<float> &dphi_dt)
    {
        ax_runner_ax650 *model;
        int len = x.size() / (2 * 80);
        if (len == 200) {
            model = &flow_estimator_200;
        } else if (len == 250) {
            model = &flow_estimator_250;
        } else if (len == 300) {
            model = &flow_estimator_300;
        } else {
            return -1;
        }

        void *p = (void *)model->get_input("x").phyAddr;
        axcl_Memcpy(p, x.data(), x.size() * sizeof(float), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);
        p = (void *)model->get_input("mask").phyAddr;
        axcl_Memcpy(p, mask.data(), mask.size() * sizeof(float), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE,
                    _attr.devid);
        p = (void *)model->get_input("t").phyAddr;
        axcl_Memcpy(p, t.data(), t.size() * sizeof(float), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);
        p = (void *)model->get_input("mu").phyAddr;
        axcl_Memcpy(p, mu.data(), mu.size() * sizeof(float), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);
        p = (void *)model->get_input("spks").phyAddr;
        axcl_Memcpy(p, spks.data(), spks.size() * sizeof(float), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE,
                    _attr.devid);
        p = (void *)model->get_input("cond").phyAddr;
        axcl_Memcpy(p, cond.data(), cond.size() * sizeof(float), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE,
                    _attr.devid);

        model->inference();

        auto &output_dphi_dt = model->get_output("y");
        if (dphi_dt.empty() || dphi_dt.size() != output_dphi_dt.nSize / sizeof(float)) {
            dphi_dt.resize(output_dphi_dt.nSize / sizeof(float));
        }
        // axcl_Memcpy(dphi_dt.data(), (void *)output_dphi_dt.phyAddr, output_dphi_dt.nSize,
        // axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, devid);
        axcl_Memcpy((void *)output_dphi_dt.pVirAddr, (void *)output_dphi_dt.phyAddr, output_dphi_dt.nSize,
                    axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, _attr.devid);
        memcpy(dphi_dt.data(), (void *)output_dphi_dt.pVirAddr, output_dphi_dt.nSize);

        return 0;
    }

    int infer_hift(std::vector<float> &mel, std::vector<float> &cache_source, std::vector<float> &tts_speech,
                   std::vector<float> &tts_source)
    {
        std::shared_ptr<BaseRunner> model_p1;
        ax_runner_ax650 *model_p2;
        int len = mel.size() / (80);

        if (len == 50 && cache_source.empty()) {
            model_p1 = hift_p1_50_first;
            model_p2 = &hift_p2_50_first;
        } else if (len == 58 && !cache_source.empty()) {
            model_p1 = hift_p1_58;
            model_p2 = &hift_p2_58;
        } else {
            ALOGE("invalid size: %d", len);
            return -1;
        }

        float *p_input = (float *)model_p1->getInputPtr(0);
        memcpy(p_input, mel.data(), mel.size() * sizeof(float));
        model_p1->inference();

        auto p_s = model_p2->get_input("s");
        memcpy(p_s.pVirAddr, (void *)model_p1->getOutputPtr(0), len * 480 * sizeof(float));
        axcl_Memcpy((void *)p_s.phyAddr, (void *)p_s.pVirAddr, len * 480 * sizeof(float),
                    axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);

        void *p = (void *)model_p2->get_input("mel").phyAddr;
        axcl_Memcpy(p, mel.data(), mel.size() * sizeof(float), axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE,
                    _attr.devid);

        if (!cache_source.empty()) {
            p = (void *)model_p2->get_input("hift_cache_source").phyAddr;
            axcl_Memcpy(p, cache_source.data(), cache_source.size() * sizeof(float),
                        axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _attr.devid);
        }

        model_p2->inference();

        auto &output_speech = model_p2->get_output("audio");
        if (tts_speech.empty() || tts_speech.size() != output_speech.nSize / sizeof(float)) {
            tts_speech.resize(output_speech.nSize / sizeof(float));
        }
        // axcl_Memcpy(tts_speech.data(), (void *)output_speech.phyAddr, output_speech.nSize,
        // axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, devid);
        axcl_Memcpy((void *)output_speech.pVirAddr, (void *)output_speech.phyAddr, output_speech.nSize,
                    axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, _attr.devid);
        memcpy(tts_speech.data(), (void *)output_speech.pVirAddr, output_speech.nSize);

        auto &output_source = model_p2->get_output(1);
        if (tts_source.empty() || tts_source.size() != output_source.nSize / sizeof(float)) {
            tts_source.resize(output_source.nSize / sizeof(float));
        }
        // axcl_Memcpy(tts_source.data(), (void *)output_source.phyAddr, output_source.nSize,
        // axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, devid);
        axcl_Memcpy((void *)output_source.pVirAddr, (void *)output_source.phyAddr, output_source.nSize,
                    axclrtMemcpyKind::AXCL_MEMCPY_DEVICE_TO_HOST, _attr.devid);
        memcpy(tts_source.data(), (void *)output_source.pVirAddr, output_source.nSize);
        return 0;
    }

    int infer_flow_decoder_solve_euler(std::vector<float> &x, std::vector<float> &mu, std::vector<float> &spks,
                                       std::vector<float> &cond, std::vector<float> &mask, std::vector<float> &mel)
    {
        int len = mu.size() / 80;

        float t  = t_span[0];
        float dt = t_span[1] - t_span[0];

        std::vector<float> x_in(2 * 80 * len, 0);
        std::vector<float> mask_in(2 * 1 * len, 0);
        std::vector<float> mu_in(2 * 80 * len, 0);
        std::vector<float> t_in(2, 0);
        std::vector<float> spks_in(2 * 80, 0);
        std::vector<float> cond_in(2 * 80 * len, 0);
        for (int step = 1; step < t_span.size(); step++) {
            memcpy(x_in.data(), x.data(), x.size() * sizeof(float));
            memcpy(x_in.data() + x.size(), x.data(), x.size() * sizeof(float));

            memcpy(mask_in.data(), mask.data(), mask.size() * sizeof(float));
            memcpy(mask_in.data() + mask.size(), mask.data(), mask.size() * sizeof(float));

            memcpy(mu_in.data(), mu.data(), mu.size() * sizeof(float));

            t_in[0] = t;
            t_in[1] = t;

            memcpy(spks_in.data(), spks.data(), spks.size() * sizeof(float));
            memcpy(cond_in.data(), cond.data(), cond.size() * sizeof(float));

            std::vector<float> dphi_dt;
            int ret = infer_flow_estimator(x_in, mask_in, t_in, mu_in, spks_in, cond_in, dphi_dt);
            if (ret != 0) {
                return ret;
            }

            for (int i = 0; i < 80 * len; i++) {
                dphi_dt[i] =
                    (1.0 + _attr.inference_cfg_rate) * dphi_dt[i] - _attr.inference_cfg_rate * dphi_dt[80 * len + i];
                x[i] = x[i] + dt * dphi_dt[i];
            }

            t = t + dt;

            if (step < t_span.size() - 1) {
                dt = t_span[step + 1] - t;
            } else {
                if (mel.empty() || mel.size() != x.size()) {
                    mel.resize(x.size());
                }
                memcpy(mel.data(), x.data(), x.size() * sizeof(float));
            }
        }

        return 0;
    }

    int infer_flow_decoder(std::vector<float> &mu, std::vector<float> &spks, std::vector<float> &cond,
                           std::vector<float> &mask, std::vector<float> &mel)
    {
        std::vector<float> z;
        z.insert(z.end(), rand_noise.begin(), rand_noise.begin() + mu.size());

        int ret = infer_flow_decoder_solve_euler(z, mu, spks, cond, mask, mel);
        return ret;
    }

    std::vector<float> infer_flow(std::vector<float> &token_embeds, std::vector<float> &prompt_feat,
                                  std::vector<float> &spk_embeds, int token_len, bool finalize)
    {
        int ret;
        int len;
        std::vector<float> mu;
        std::vector<float> spks;
        std::vector<float> cond;

        ret = infer_flow_encoder(token_embeds, prompt_feat, spk_embeds, token_len, finalize, mu, spks, cond);
        if (ret != 0) {
            return std::vector<float>{};
        }

        len = mu.size() / 80;

        std::vector<float> mask(len, 1.0);
        std::vector<float> all_mel;

        ret = infer_flow_decoder(mu, spks, cond, mask, all_mel);
        if (ret != 0) {
            return std::vector<float>{};
        }

        int len_mel1 = prompt_feat.size() / 80;
        int len_mel2 = all_mel.size() / 80 - len_mel1;

        std::vector<float> mel(len_mel2 * 80, 0);
        auto result = slice_3d_last_dim_from<float>(all_mel, 1, 80, all_mel.size() / 80, len_mel1);

        return result;
    }

    void fade_in_out(std::vector<float> &fade_in_mel_data, const std::vector<float> &fade_out_mel_data,
                     const std::vector<float> &window)
    {
        // --- Constants based on window = np.hamming(2 * 8 * 480) ---
        const size_t WINDOW_SIZE     = 2 * 8 * 480;      // 7680
        const size_t MEL_OVERLAP_LEN = WINDOW_SIZE / 2;  // 3840
        // dim0 is implicitly 1 for both inputs
        size_t dim1_in  = fade_in_mel_data.size();
        size_t dim1_out = fade_out_mel_data.size();
        // --- Input Validation ---
        // For 2D arrays [1, L], the 1D vector size is just L.

        if (window.size() != WINDOW_SIZE) {
            throw std::invalid_argument("window size (" + std::to_string(window.size()) +
                                        ") does not match expected size (7680).");
        }
        // Check if input arrays have enough elements for the overlap
        if (dim1_in < MEL_OVERLAP_LEN) {
            throw std::invalid_argument("fade_in_mel_data's column count (" + std::to_string(dim1_in) +
                                        ") is smaller than mel_overlap_len (" + std::to_string(MEL_OVERLAP_LEN) + ").");
        }
        if (dim1_out < MEL_OVERLAP_LEN) {
            throw std::invalid_argument("fade_out_mel_data's column count (" + std::to_string(dim1_out) +
                                        ") is smaller than mel_overlap_len (" + std::to_string(MEL_OVERLAP_LEN) + ").");
        }

        // --- Perform Fade In/Out ---
        // Since dim0=1, we only have one "row" to process.
        // Iterate through the overlapping elements in the column dimension.
        for (size_t i = 0; i < MEL_OVERLAP_LEN; ++i) {
            // Indices are simply 'i' for the start of fade_in_mel
            // and 'dim1_out - MEL_OVERLAP_LEN + i' for the end of fade_out_mel
            const size_t in_idx  = i;
            const size_t out_idx = dim1_out - MEL_OVERLAP_LEN + i;

            // Perform the weighted sum: result = in_val * win_in + out_val * win_out
            // in_val = fade_in_mel_data[in_idx]
            // out_val = fade_out_mel_data[out_idx]
            // win_in = window[i]
            // win_out = window[MEL_OVERLAP_LEN + i]
            fade_in_mel_data[in_idx] =
                fade_in_mel_data[in_idx] * window[i] + fade_out_mel_data[out_idx] * window[MEL_OVERLAP_LEN + i];
        }
        // fade_in_mel_data is now modified in-place with the faded result.
    }

    void clear()
    {
        std::unordered_map<std::string, std::vector<float>>().swap(hift_cache_dict);
    }

    std::vector<float> infer(std::vector<int> &text_speech_token, std::vector<float> &prompt_speech_embeds,
                             std::vector<float> &prompt_feat, std::vector<float> &spk_embeds, int token_offset,
                             bool finalize)
    {
        int ret = 0;
        std::vector<float> speech_embeds(text_speech_token.size() * _attr.flow_embed_size + prompt_speech_embeds.size(),
                                         0.0f);
        std::vector<unsigned short> speech_embeds_one(_attr.flow_embed_size, 0);

        memcpy(speech_embeds.data(), prompt_speech_embeds.data(), prompt_speech_embeds.size() * sizeof(float));

        for (size_t i = 0; i < text_speech_token.size(); i++) {
            flow_embed_selector.getByIndex(text_speech_token[i], speech_embeds_one.data());

            for (int j = 0; j < _attr.flow_embed_size; j++) {
                unsigned int proc = speech_embeds_one[j] << 16;
                speech_embeds[prompt_speech_embeds.size() + i * _attr.flow_embed_size + j] =
                    *reinterpret_cast<float *>(&proc);
            }
        }

        std::vector<float> mel;

        mel = infer_flow(speech_embeds, prompt_feat, spk_embeds, text_speech_token.size(), finalize);

        std::vector<float> tts_mel;
        int neg_offset = 0, start;
        if (finalize) {
            neg_offset = token_offset * _attr.token_mel_ratio - mel.size() / 80;
            start      = -_attr.token_hop_len * _attr.token_mel_ratio;
        } else {
            start = std::min(int(token_offset / _attr.token_hop_len), _attr.max_infer_chunk_num - 1) *
                    _attr.token_hop_len * _attr.token_mel_ratio;
        }

        tts_mel = slice_3d_last_dim_from<float>(mel, 1, 80, mel.size() / 80, start);

        std::vector<float> hift_cache_source;
        std::vector<float> tts_mel1;
        std::vector<float> speech, source, tts_speech;
        if (!hift_cache_dict.empty()) {
            auto hift_cache_mel = hift_cache_dict["mel"];
            hift_cache_source   = hift_cache_dict["source"];
            tts_mel1 = concat_3d_dim2<float>(hift_cache_mel, 1, 80, hift_cache_mel.size() / 80, tts_mel, 1, 80,
                                             tts_mel.size() / 80);
        } else {
            tts_mel1 = tts_mel;
        }
        ret = infer_hift(tts_mel1, hift_cache_source, speech, source);

        if (ret != 0) {
            ALOGE("failed");
            return std::vector<float>{};
        }

        if (!finalize) {
            if (!hift_cache_dict.empty()) {
                fade_in_out(speech, hift_cache_dict["speech"], speech_window);
            }

            hift_cache_dict["mel"] =
                slice_3d_last_dim_from<float>(tts_mel1, 1, 80, tts_mel1.size() / 80, -_attr.mel_cache_len);

            int offset = speech.size();
            if (speech.size() > _attr.source_cache_len) {
                offset = _attr.source_cache_len;
            }

            hift_cache_dict["source"].assign(source.end() - offset, source.end());
            hift_cache_dict["speech"].assign(speech.end() - offset, speech.end());
            tts_speech.assign(speech.begin(), speech.end() - offset);

        } else {
            if (speech.size() < _attr.source_cache_len) {
                tts_speech.assign(speech.begin(), speech.end());
            } else if (-neg_offset * 480 >= _attr.source_cache_len) {
                tts_speech.assign(speech.end() + neg_offset * 480, speech.end());

                if (!hift_cache_dict.empty()) {
                    fade_in_out(tts_speech, hift_cache_dict["speech"], speech_window);
                }
            } else {
                tts_speech.assign(speech.end() - _attr.source_cache_len, speech.end());

                if (!hift_cache_dict.empty()) {
                    fade_in_out(tts_speech, hift_cache_dict["speech"], speech_window);
                }

                int offset = speech.size() + neg_offset * 480 - (speech.size() - _attr.source_cache_len);
                tts_speech.assign(tts_speech.begin() + offset, tts_speech.end());
            }
        }

        return tts_speech;
    }
};