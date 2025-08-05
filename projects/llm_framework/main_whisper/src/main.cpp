/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include <axcl.h>
#include "AudioFile.h"
#include "opencc.h"
#include <librosa/librosa.h>

#include "middleware/axcl_runtime_runner.hpp"

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <base64.h>
#include <fstream>
#include <vector>
#include <string.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "subprocess.h"
#define BUFFER_IMPLEMENTATION
#include "libs/buffer.h"

using namespace StackFlows;

int main_exit_flage = 0;

static void __sigint(int iSigNo)
{
    SLOGW("llm_whisper will be exit!");
    main_exit_flage = 1;
}

static std::string base_model_path_;
static std::string base_model_config_path_;
const char *CONFIG_FILE_DEFAULT = "/usr/local/axcl/axcl.json";

typedef struct {
    std::string encoder;
    std::string decoder_main;
    std::string decoder_loop;
    std::string positional_embedding;
    std::string tokens;
    std::vector<std::string> token_tables;
    std::string model_type;
    std::string language;
    std::string t2s;
    int whisper_sample_rate   = 16000;
    int whisper_n_fft         = 400;
    int awake_delay           = 1;
    int whisper_hop_length    = 160;
    int whisper_chunk_size    = 30;
    int whisper_n_mels        = 400;
    int whisper_sot           = 50258;
    int whisper_eot           = 50257;
    int whisper_blank         = 220;
    int whisper_no_timestamps = 50363;
    int whisper_no_speech     = 50362;
    int whisper_translate     = 50358;
    int whisper_transcribe    = 50359;
    int whisper_vocab_size    = 51865;
    int whisper_n_text_ctx    = 448;
    float neg_inf             = -std::numeric_limits<float>::infinity();
} whisper_config;

typedef std::function<void(const std::string &data, bool finish)> task_callback_t;

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

class llm_task {
private:
    std::unique_ptr<middleware::runner> encoder_;
    std::unique_ptr<middleware::runner> decoder_main_;
    std::unique_ptr<middleware::runner> decoder_loop_;
    whisper_config mode_config_;

public:
    std::vector<float> positional_embedding;
    std::string model_;
    std::string response_format_;
    std::vector<std::string> inputs_;
    std::string language_;
    bool enoutput_;
    bool enstream_;
    bool ensleep_;
    std::atomic_bool superior_flage_;
    std::atomic_bool audio_flage_;
    std::atomic_bool awake_flage_;
    std::atomic_bool endpoint_flage_;
    std::string superior_id_;
    static int ax_init_flage_;
    task_callback_t out_callback_;
    int awake_delay_       = 50;
    int delay_audio_frame_ = 1000;
    buffer_t *pcmdata;

    std::function<void(void)> pause;

    std::unique_ptr<middleware::runner> load_runner(const std::string &model_path)
    {
        std::unique_ptr<middleware::runner> runner = std::make_unique<middleware::runtime_runner>();

        if (!runner->init(CONFIG_FILE_DEFAULT, 0, 0)) {
            fprintf(stderr, "[ERROR] Init failed.\n");
            return nullptr;
        }

        if (!runner->load(model_path)) {
            fprintf(stderr, "[ERROR] Loading model {%s} failed.\n", model_path.c_str());
            return nullptr;
        }

        if (!runner->prepare(true, true, 0, 0)) {
            fprintf(stderr, "[ERROR] Prepare for model {%s} failed.\n", model_path.c_str());
            return nullptr;
        }

        return std::move(runner);
    }

    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            model_           = config_body.at("model");
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");
            language_        = config_body.at("language");
            if (config_body.contains("input")) {
                if (config_body["input"].is_string()) {
                    inputs_.push_back(config_body["input"].get<std::string>());
                } else if (config_body["input"].is_array()) {
                    for (auto _in : config_body["input"]) {
                        inputs_.push_back(_in.get<std::string>());
                    }
                }
            } else
                throw std::string("error");
        } catch (...) {
            SLOGE("setup config_body error");
            return true;
        }
        enstream_ = response_format_.find("stream") == std::string::npos ? false : true;
        return false;
    }

    std::vector<int> WHISPER_LANG_CODES{
        50273, 50303, 50288, 50261, 50342, 50299, 50330, 50302, 50336, 50267, 50287, 50292, 50294, 50323, 50348,
        50291, 50317, 50326, 50289, 50356, 50290, 50282, 50347, 50331, 50354, 50264, 50333, 50296, 50339, 50318,
        50305, 50293, 50280, 50322, 50312, 50306, 50353, 50285, 50275, 50340, 50278, 50268, 50337, 50316, 50266,
        50307, 50310, 50338, 50334, 50313, 50351, 50260, 50344, 50283, 50327, 50272, 50324, 50276, 50281, 50301,
        50332, 50300, 50309, 50343, 50349, 50335, 50320, 50259, 50284, 50304, 50277, 50311, 50319, 50314, 50352,
        50328, 50286, 50274, 50329, 50270, 50269, 50350, 50263, 50345, 50298, 50279, 50297, 50262, 50315, 50321,
        50308, 50355, 50265, 50346, 50295, 50271, 50357, 50341, 50325};

    std::vector<std::string> WHISPER_LANG_NAMES{
        "sv", "sr", "no", "de", "nn", "te", "be",  "bn", "lo", "pt", "ta", "bg", "la", "km", "tl", "hr", "sq",
        "so", "th", "jw", "ur", "ms", "bo", "tg",  "ha", "ko", "gu", "ml", "ht", "sw", "sl", "lt", "uk", "si",
        "hy", "kn", "ln", "da", "id", "ps", "vi",  "tr", "uz", "kk", "ja", "et", "eu", "fo", "am", "ne", "tt",
        "zh", "sa", "cs", "af", "ar", "sn", "hi",  "el", "lv", "sd", "fa", "br", "mt", "mg", "yi", "mr", "en",
        "ro", "az", "fi", "is", "gl", "mn", "haw", "oc", "hu", "it", "ka", "ca", "pl", "as", "ru", "lb", "sk",
        "he", "cy", "es", "bs", "pa", "mk", "ba",  "fr", "my", "mi", "nl", "su", "tk", "yo"};

    std::unordered_map<std::string, int> WHISPER_N_TEXT_STATE_MAP{{"tiny", 384}, {"base", 512}, {"small", 768}};

    std::vector<int32_t> SOT_SEQUENCE{mode_config_.whisper_sot, 50260, mode_config_.whisper_transcribe,
                                      mode_config_.whisper_no_timestamps};

    void supress_tokens(std::vector<float> &logits, bool is_initial)
    {
        mode_config_.neg_inf = -std::numeric_limits<float>::infinity();
        if (is_initial) {
            logits[mode_config_.whisper_eot]   = mode_config_.neg_inf;
            logits[mode_config_.whisper_blank] = mode_config_.neg_inf;
        }

        logits[mode_config_.whisper_no_timestamps] = mode_config_.neg_inf;
        logits[mode_config_.whisper_sot]           = mode_config_.neg_inf;
        logits[mode_config_.whisper_no_speech]     = mode_config_.neg_inf;
        logits[mode_config_.whisper_translate]     = mode_config_.neg_inf;
    }

    int argmax(const std::vector<float> &logits)
    {
        auto max_iter = std::max_element(logits.begin(), logits.end());
        return std::distance(logits.begin(), max_iter);  // absolute index of max
    }

    int detect_language(const std::string &language)
    {
        int i = 51;  // zh
        for (int n = 0; n < WHISPER_LANG_CODES.size(); n++) {
            if (language == WHISPER_LANG_NAMES[n]) {
                i = n;
                break;
            }
        }

        return WHISPER_LANG_CODES[i];
    }

    double get_current_time()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }

    bool is_valid_utf8(const std::string &str)
    {
        int bytes = 0;
        for (unsigned char c : str) {
            if (bytes == 0) {
                if ((c >> 5) == 0b110)
                    bytes = 1;
                else if ((c >> 4) == 0b1110)
                    bytes = 2;
                else if ((c >> 3) == 0b11110)
                    bytes = 3;
                else if ((c >> 7))
                    return false;
            } else {
                if ((c >> 6) != 0b10) return false;
                bytes--;
            }
        }
        return bytes == 0;
    }

    void fix_utf8_string(std::string &s)
    {
        while (!s.empty() && !is_valid_utf8(s)) {
            s.pop_back();
        }
    }

    int load_model(const nlohmann::json &config_body)
    {
        if (parse_config(config_body)) {
            return -1;
        }
        nlohmann::json file_body;
        std::list<std::string> config_file_paths =
            get_config_file_paths(base_model_path_, base_model_config_path_, model_);
        // Compatible operation
        if (model_ == "whisper-tiny")
            config_file_paths = get_config_file_paths(base_model_path_, base_model_config_path_, "whisper-tiny");
        else if (model_ == "whisper-base")
            config_file_paths = get_config_file_paths(base_model_path_, base_model_config_path_, "whisper-base");
        try {
            for (auto file_name : config_file_paths) {
                std::ifstream config_file(file_name);
                if (!config_file.is_open()) {
                    SLOGW("config file :%s miss", file_name.c_str());
                    continue;
                }
                SLOGI("config file :%s read", file_name.c_str());
                config_file >> file_body;
                config_file.close();
                break;
            }
            if (file_body.empty()) {
                SLOGE("all config file miss");
                return -2;
            }
            std::string base_model = base_model_path_ + model_ + "/";
            SLOGI("base_model %s", base_model.c_str());
            CONFIG_AUTO_SET(file_body["mode_param"], tokens);
            CONFIG_AUTO_SET(file_body["mode_param"], positional_embedding);
            CONFIG_AUTO_SET(file_body["mode_param"], encoder);
            CONFIG_AUTO_SET(file_body["mode_param"], decoder_main);
            CONFIG_AUTO_SET(file_body["mode_param"], decoder_loop);
            CONFIG_AUTO_SET(file_body["mode_param"], language);
            CONFIG_AUTO_SET(file_body["mode_param"], t2s);
            CONFIG_AUTO_SET(file_body["mode_param"], model_type);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_sample_rate);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_n_fft);
            CONFIG_AUTO_SET(file_body["mode_param"], awake_delay);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_hop_length);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_chunk_size);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_n_mels);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_sot);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_eot);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_blank);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_no_timestamps);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_no_speech);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_translate);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_transcribe);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_vocab_size);
            CONFIG_AUTO_SET(file_body["mode_param"], whisper_n_text_ctx);
            mode_config_.tokens               = base_model + mode_config_.tokens;
            mode_config_.positional_embedding = base_model + mode_config_.positional_embedding;
            mode_config_.encoder              = base_model + mode_config_.encoder;
            mode_config_.decoder_main         = base_model + mode_config_.decoder_main;
            mode_config_.decoder_loop         = base_model + mode_config_.decoder_loop;
            mode_config_.t2s                  = base_model + mode_config_.t2s;
            if (config_body.contains("awake_delay"))
                awake_delay_ = config_body["awake_delay"].get<int>();
            else if (file_body["mode_param"].contains("awake_delay"))
                awake_delay_ = file_body["mode_param"]["awake_delay"];
            int WHISPER_N_TEXT_STATE = WHISPER_N_TEXT_STATE_MAP[mode_config_.model_type];
            positional_embedding.resize(mode_config_.whisper_n_text_ctx * WHISPER_N_TEXT_STATE);
            FILE *fp = fopen(mode_config_.positional_embedding.c_str(), "rb");
            if (!fp) {
                SLOGE("Open %s failed!\n", mode_config_.positional_embedding.c_str());
                return -3;
            }
            fread(positional_embedding.data(), sizeof(float), mode_config_.whisper_n_text_ctx * WHISPER_N_TEXT_STATE,
                  fp);
            fclose(fp);

            std::ifstream ifs(mode_config_.tokens);
            if (!ifs.is_open()) {
                fprintf(stderr, "Can NOT open %s\n", mode_config_.tokens.c_str());
                return false;
            }
            std::string line;
            while (std::getline(ifs, line)) {
                size_t i = line.find(' ');
                mode_config_.token_tables.push_back(line.substr(0, i));
            }

            encoder_      = load_runner(mode_config_.encoder);
            decoder_main_ = load_runner(mode_config_.decoder_main);
            decoder_loop_ = load_runner(mode_config_.decoder_loop);
        } catch (...) {
            SLOGE("config false");
            return -6;
        }
        return 0;
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }

    void sys_pcm_on_data(const std::string &raw)
    {
        static int count = 0;
        double start, end;
        double start_all, end_all;
        if (count < delay_audio_frame_) {
            buffer_write_char(pcmdata, raw.c_str(), raw.length());
            count++;
            if (endpoint_flage_) return;
        }
        endpoint_flage_ = true;
        if (delay_audio_frame_ == 0) buffer_resize(pcmdata, 0);
        buffer_write_char(pcmdata, raw.c_str(), raw.length());
        buffer_position_set(pcmdata, 0);
        count = 0;
        std::vector<float> floatSamples;
        {
            int16_t audio_val;
            while (buffer_read_u16(pcmdata, (unsigned short *)&audio_val, 1)) {
                float normalizedSample = (float)audio_val / INT16_MAX;
                floatSamples.push_back(normalizedSample);
            }
        }
        buffer_position_set(pcmdata, 0);

        if (WHISPER_N_TEXT_STATE_MAP.find(mode_config_.model_type) == WHISPER_N_TEXT_STATE_MAP.end()) {
            fprintf(stderr, "Can NOT find n_text_state for model_type: %s\n", mode_config_.model_type.c_str());
            return;
        }

        int WHISPER_N_TEXT_STATE = WHISPER_N_TEXT_STATE_MAP[mode_config_.model_type];

        auto mel = librosa::Feature::melspectrogram(
            floatSamples, mode_config_.whisper_sample_rate, mode_config_.whisper_n_fft, mode_config_.whisper_hop_length,
            "hann", true, "reflect", 2.0f, mode_config_.whisper_n_mels, 0.0f, mode_config_.whisper_sample_rate / 2.0f);
        int n_mel = mel.size();
        int n_len = mel[0].size();

        // clamping and normalization
        double mmax = -1e20;
        for (int i = 0; i < mode_config_.whisper_n_mels; i++) {
            for (int n = 0; n < n_len; n++) {
                mel[i][n] = std::log10(std::max(mel[i][n], 1e-10f));

                if (mel[i][n] > mmax) {
                    mmax = mel[i][n];
                }
            }
        }

        for (int i = 0; i < mode_config_.whisper_n_mels; i++) {
            for (int n = 0; n < n_len; n++) {
                mel[i][n] = (std::max(mel[i][n], (float)(mmax - 8.0)) + 4.0) / 4.0;
                mel[i].resize(3000);
            }
        }

        n_len = mel[0].size();

        int offset = 0;
        std::vector<float> logits(mode_config_.whisper_vocab_size);
        int max_token_id = -1;
        std::vector<int> results;
        std::vector<int> tokens(1);
        bool is_broke = false;

        std::vector<float> n_layer_cross_k(encoder_->get_output_size(0) / sizeof(float));
        std::vector<float> n_layer_cross_v(encoder_->get_output_size(1) / sizeof(float));

        std::vector<float> decoder_main_logits(4 * mode_config_.whisper_vocab_size);
        std::vector<float> n_layer_self_k_cache(decoder_main_->get_output_size(1) / sizeof(float));
        std::vector<float> n_layer_self_v_cache(decoder_main_->get_output_size(2) / sizeof(float));

        std::vector<float> continous_mel(mode_config_.whisper_n_mels * n_len);
        for (int i = 0; i < n_mel; i++) {
            memcpy(continous_mel.data() + i * n_len, mel[i].data(), sizeof(float) * n_len);
        }

        start     = get_current_time();
        start_all = get_current_time();
        axclrtMemcpy(encoder_->get_input_pointer(0), continous_mel.data(), sizeof(float) * continous_mel.size(), AXCL_MEMCPY_HOST_TO_DEVICE);
        int ret = encoder_->run(false);
        if (ret) {
            SLOGE("encoder run failed!");
            return;
        }
        end = get_current_time();
        SLOGI("Encoder run take %.2f ms\n", (end - start));

        // detect language
        SOT_SEQUENCE[1] = detect_language(language_);

        // decoder_main
        start = get_current_time();
        axclrtMemcpy(decoder_main_->get_input_pointer(0), SOT_SEQUENCE.data(), sizeof(int) * SOT_SEQUENCE.size(), AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(decoder_main_->get_input_pointer(1), encoder_->get_output_pointer(0), decoder_main_->get_input_size(1), AXCL_MEMCPY_DEVICE_TO_DEVICE);
        axclrtMemcpy(decoder_main_->get_input_pointer(2), encoder_->get_output_pointer(1), decoder_main_->get_input_size(2), AXCL_MEMCPY_DEVICE_TO_DEVICE);
        ret = decoder_main_->run(false);
        if (ret) {
            SLOGE("decoder_main run failed!");
            return;
        }
        axclrtMemcpy(decoder_main_logits.data(), decoder_main_->get_output_pointer(0), sizeof(float) * decoder_main_logits.size(), AXCL_MEMCPY_DEVICE_TO_HOST);

        end = get_current_time();

        offset += SOT_SEQUENCE.size();

        std::copy(decoder_main_logits.begin() + 3 * mode_config_.whisper_vocab_size, decoder_main_logits.end(),
                  logits.begin());
        supress_tokens(logits, true);

        max_token_id = argmax(logits);
        SLOGI("First token: %d \t take %.2fms\n", max_token_id, (end - start));
        mode_config_.neg_inf = -std::numeric_limits<float>::infinity();
        std::vector<float> mask(mode_config_.whisper_n_text_ctx);
        for (int n = 0; n < mode_config_.whisper_n_text_ctx - offset - 1; n++) {
            mask[n] = mode_config_.neg_inf;
        }

        axclrtMemcpy(decoder_loop_->get_input_pointer(1), decoder_main_->get_output_pointer(1), decoder_loop_->get_input_size(1), AXCL_MEMCPY_DEVICE_TO_DEVICE);
        axclrtMemcpy(decoder_loop_->get_input_pointer(2), decoder_main_->get_output_pointer(2), decoder_loop_->get_input_size(2), AXCL_MEMCPY_DEVICE_TO_DEVICE);
        axclrtMemcpy(decoder_loop_->get_input_pointer(3), encoder_->get_output_pointer(0), decoder_loop_->get_input_size(3), AXCL_MEMCPY_DEVICE_TO_DEVICE);
        axclrtMemcpy(decoder_loop_->get_input_pointer(4), encoder_->get_output_pointer(1), decoder_loop_->get_input_size(4), AXCL_MEMCPY_DEVICE_TO_DEVICE);

        for (int i = 0; i < mode_config_.whisper_n_text_ctx - SOT_SEQUENCE.size(); i++) {
            if (max_token_id == mode_config_.whisper_eot) {
                is_broke = true;
                break;
            }

            results.push_back(max_token_id);
            tokens[0] = results.back();

            // inference
            start = get_current_time();
            axclrtMemcpy(decoder_loop_->get_input_pointer(0), tokens.data(), sizeof(int) * tokens.size(), AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(decoder_loop_->get_input_pointer(5), positional_embedding.data() + offset * WHISPER_N_TEXT_STATE, decoder_loop_->get_input_size(5), AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(decoder_loop_->get_input_pointer(6), mask.data(), sizeof(float) * mask.size(), AXCL_MEMCPY_HOST_TO_DEVICE);

            ret = decoder_loop_->run(false);
            if (ret) {
                SLOGE("decoder_loop run failed!\n");
                return;
            }

            axclrtMemcpy(decoder_loop_->get_input_pointer(1), decoder_loop_->get_output_pointer(1), sizeof(float) * n_layer_self_k_cache.size(), AXCL_MEMCPY_DEVICE_TO_DEVICE);
            axclrtMemcpy(decoder_loop_->get_input_pointer(2), decoder_loop_->get_output_pointer(2), sizeof(float) * n_layer_self_v_cache.size(), AXCL_MEMCPY_DEVICE_TO_DEVICE);
            axclrtMemcpy(logits.data(), decoder_loop_->get_output_pointer(0), sizeof(float) * logits.size(), AXCL_MEMCPY_DEVICE_TO_HOST);

            offset += 1;
            mask[mode_config_.whisper_n_text_ctx - offset - 1] = 0;

            supress_tokens(logits, false);
            max_token_id = argmax(logits);
            end          = get_current_time();

            SLOGI("Next Token: %d \t take %.2fms\n", max_token_id, (end - start));
        }

        end_all = get_current_time();
        SLOGI("All take %.2f ms\n", (end_all - start_all));

        std::string s;
        for (const auto i : results) {
            char str[1024];
            base64_decode((const uint8 *)mode_config_.token_tables[i].c_str(),
                          (uint32)mode_config_.token_tables[i].size(), str);
            s += str;
        }
        fix_utf8_string(s);
        if (mode_config_.language == "en" || mode_config_.language == "ja") {
            if (out_callback_) out_callback_(s, true);
        } else {
            const opencc::SimpleConverter converter(mode_config_.t2s.c_str());
            std::string simple_str = converter.Convert(s);
            if ((!simple_str.empty()) && out_callback_) {
                out_callback_(simple_str, true);
            }
        }

        if (ensleep_) {
            if (pause) pause();
        }
    }

    void kws_awake()
    {
        awake_flage_ = true;
    }

    void _ax_init()
    {
        if (!ax_init_flage_) {
            int ret = axclInit(nullptr);
            if (0 != ret) {
                fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
            }
        }
        ax_init_flage_++;
    }

    void _ax_deinit()
    {
        if (ax_init_flage_ > 0) {
            --ax_init_flage_;
            if (!ax_init_flage_) {
                axclFinalize();
            }
        }
    }

    llm_task(const std::string &workid)
    {
        ensleep_     = false;
        awake_flage_ = false;
        pcmdata      = buffer_create();
        _ax_init();
    }

    void start()
    {
    }

    void stop()
    {
    }

    ~llm_task()
    {
        stop();
        if (encoder_) encoder_->final();
        if (decoder_main_) decoder_main_->final();
        if (decoder_loop_) decoder_loop_->final();
        _ax_deinit();
        buffer_destroy(pcmdata);
    }
};

int llm_task::ax_init_flage_ = 0;
#undef CONFIG_AUTO_SET

class llm_whisper : public StackFlow {
private:
    int task_count_;
    std::string audio_url_;
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    enum { EVENT_LOAD_CONFIG = EVENT_EXPORT + 1, EVENT_TASK_PAUSE };

    llm_whisper() : StackFlow("whisper")
    {
        task_count_ = 1;
        event_queue_.appendListener(EVENT_TASK_PAUSE,
                                    std::bind(&llm_whisper::_task_pause, this, std::placeholders::_1));
    }

    void task_output(const std::weak_ptr<llm_task> llm_task_obj_weak,
                     const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &data, bool finish)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::string tmp_msg1;
        const std::string *next_data = &data;
        if (finish) {
            tmp_msg1  = data + ".";
            next_data = &tmp_msg1;
        }
        if (llm_channel->enstream_) {
            static int count = 0;
            nlohmann::json data_body;
            data_body["index"]  = count++;
            data_body["delta"]  = (*next_data);
            data_body["finish"] = finish;
            if (finish) count = 0;
            SLOGI("send stream:%s", next_data->c_str());
            llm_channel->send(llm_task_obj->response_format_, data_body, LLM_NO_ERROR);
        } else if (finish) {
            SLOGI("send utf-8:%s", next_data->c_str());
            llm_channel->send(llm_task_obj->response_format_, (*next_data), LLM_NO_ERROR);
        }
    }

    int decode_wav(const std::string &in, std::string &out)
    {
        int post = 0;
        if (in.length() > 10)
            for (int i = 0; i < in.length() - 4; i++) {
                if ((in[i] == 'd') && (in[i + 1] == 'a') && (in[i + 2] == 't') && (in[i + 3] == 'a')) {
                    post = i + 8;
                    break;
                }
            }
        if (post > 0) {
            out = std::string((char *)(in.c_str() + post), in.length() - post);
            return in.length() - post;
        } else {
            return 0;
        }
        return 0;
    }

    int decode_mp3(const std::string &in, std::string &out)
    {
        return 0;
    }

    void task_user_data(const std::weak_ptr<llm_task> llm_task_obj_weak,
                        const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                        const std::string &data)
    {
        nlohmann::json error_body;
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            error_body["code"]    = -11;
            error_body["message"] = "Model run failed.";
            send("None", "None", error_body, unit_name_);
            return;
        }
        std::string tmp_msg1;
        const std::string *next_data = &data;
        int ret;
        if (object.find("stream") != std::string::npos) {
            static std::unordered_map<int, std::string> stream_buff;
            try {
                if (decode_stream(data, tmp_msg1, stream_buff)) {
                    return;
                };
            } catch (...) {
                stream_buff.clear();
                error_body["code"]    = -25;
                error_body["message"] = "Stream data index error.";
                send("None", "None", error_body, unit_name_);
                return;
            }
            next_data = &tmp_msg1;
        }
        std::string tmp_msg2;
        if (object.find("base64") != std::string::npos) {
            ret = decode_base64((*next_data), tmp_msg2);
            if (ret == -1) {
                error_body["code"]    = -23;
                error_body["message"] = "Base64 decoding error.";
                send("None", "None", error_body, unit_name_);
                return;
            }
            next_data = &tmp_msg2;
        }
        std::string tmp_msg3;
        if (object.find("wav") != std::string::npos) {
            ret = decode_wav((*next_data), tmp_msg3);
            if (!ret) {
                return;
            }
            next_data = &tmp_msg3;
        }
        std::string tmp_msg4;
        if (object.find("mp3") != std::string::npos) {
            ret = decode_mp3((*next_data), tmp_msg4);
            if (!ret) {
                return;
            }
            next_data = &tmp_msg4;
        }
        llm_task_obj->sys_pcm_on_data((*next_data));
    }

    void _task_pause(const std::shared_ptr<void> &arg)
    {
        std::shared_ptr<std::string> work_id = std::static_pointer_cast<std::string>(arg);
        int work_id_num                      = sample_get_work_id_num(*work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            return;
        }
        auto llm_task_obj = llm_task_[work_id_num];
        auto llm_channel  = get_channel(work_id_num);
        if (llm_task_obj->audio_flage_) {
            if (!audio_url_.empty()) llm_channel->stop_subscriber(audio_url_);
            llm_task_obj->audio_flage_ = false;
        }
    }

    void task_pause(const std::string &work_id, const std::string &data)
    {
        event_queue_.enqueue(EVENT_TASK_PAUSE, std::make_shared<std::string>(work_id));
    }

    void task_work(const std::weak_ptr<llm_task> llm_task_obj_weak,
                   const std::weak_ptr<llm_channel_obj> llm_channel_weak)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        llm_task_obj->kws_awake();
        if ((!audio_url_.empty()) && (llm_task_obj->audio_flage_ == false)) {
            std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                _llm_task_obj.lock()->sys_pcm_on_data(raw->string());
            });
            llm_task_obj->audio_flage_ = true;
        }
    }

    void kws_awake(const std::weak_ptr<llm_task> llm_task_obj_weak,
                   const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                   const std::string &data)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(llm_task_obj->awake_delay_));
        task_work(llm_task_obj, llm_channel);
    }

    void vad_endpoint(const std::weak_ptr<llm_task> llm_task_obj_weak,
                      const std::weak_ptr<llm_channel_obj> llm_channel_weak, const std::string &object,
                      const std::string &data)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        if (data == "true" || data == "false") {
            llm_task_obj->endpoint_flage_ = (data == "true");
        }
    }

    void work(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_whisper::work:%s", data.c_str());

        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        task_work(llm_task_[work_id_num], get_channel(work_id_num));
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    void pause(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_whisper::pause:%s", data.c_str());

        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        task_pause(work_id, "");
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        if ((llm_task_channel_.size() - 1) == task_count_) {
            error_body["code"]    = -21;
            error_body["message"] = "task full";
            send("None", "None", error_body, "whisper");
            return -1;
        }

        int work_id_num   = sample_get_work_id_num(work_id);
        auto llm_channel  = get_channel(work_id);
        auto llm_task_obj = std::make_shared<llm_task>(work_id);

        nlohmann::json config_body;
        try {
            config_body = nlohmann::json::parse(data);
        } catch (...) {
            SLOGE("setup json format error.");
            error_body["code"]    = -2;
            error_body["message"] = "json format error.";
            send("None", "None", error_body, "whisper");
            return -2;
        }
        int ret = llm_task_obj->load_model(config_body);
        if (ret == 0) {
            llm_channel->set_output(llm_task_obj->enoutput_);
            llm_channel->set_stream(llm_task_obj->enstream_);
            llm_task_obj->pause = std::bind(&llm_whisper::task_pause, this, work_id, "");
            llm_task_obj->set_output(std::bind(&llm_whisper::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));

            for (const auto input : llm_task_obj->inputs_) {
                if (input.find("sys") != std::string::npos) {
                    audio_url_                            = unit_call("audio", "cap", input);
                    std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
                    llm_channel->subscriber(audio_url_,
                                            [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                                                _llm_task_obj.lock()->sys_pcm_on_data(raw->string());
                                            });
                    llm_task_obj->audio_flage_ = true;
                } else if (input.find("whisper") != std::string::npos) {
                    if (input.find("stream.base64") != std::string::npos) llm_task_obj->delay_audio_frame_ = 0;
                    llm_channel->subscriber_work_id(
                        "", std::bind(&llm_whisper::task_user_data, this, std::weak_ptr<llm_task>(llm_task_obj),
                                      std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                      std::placeholders::_2));
                } else if (input.find("kws") != std::string::npos) {
                    llm_task_obj->ensleep_ = true;
                    task_pause(work_id, "");
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_whisper::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                } else if (input.find("vad") != std::string::npos) {
                    llm_task_obj->endpoint_flage_ = true;
                    // task_pause(work_id, "");
                    llm_channel->subscriber_work_id(
                        input, std::bind(&llm_whisper::vad_endpoint, this, std::weak_ptr<llm_task>(llm_task_obj),
                                         std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                         std::placeholders::_2));
                }
            }
            llm_task_[work_id_num] = llm_task_obj;
            SLOGI("load_mode success");
            send("None", "None", LLM_NO_ERROR, work_id);
            return 0;
        } else {
            SLOGE("load_mode Failed");
            error_body["code"]    = -5;
            error_body["message"] = "Model loading failed.";
            send("None", "None", error_body, "asr");
            return -1;
        }
    }

    void link(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_whisper::link:%s", data.c_str());
        int ret = 1;
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        auto llm_channel  = get_channel(work_id);
        auto llm_task_obj = llm_task_[work_id_num];
        if (data.find("sys") != std::string::npos) {
            if (audio_url_.empty()) audio_url_ = unit_call("audio", "cap", data);
            std::weak_ptr<llm_task> _llm_task_obj = llm_task_obj;
            llm_channel->subscriber(audio_url_, [_llm_task_obj](pzmq *_pzmq, const std::shared_ptr<pzmq_data> &raw) {
                _llm_task_obj.lock()->sys_pcm_on_data(raw->string());
            });
            llm_task_obj->audio_flage_ = true;
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("kws") != std::string::npos) {
            llm_task_obj->ensleep_ = true;
            ret                    = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_whisper::kws_awake, this, std::weak_ptr<llm_task>(llm_task_obj),
                                             std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
            llm_task_obj->inputs_.push_back(data);
        } else if (data.find("vad") != std::string::npos) {
            llm_task_obj->endpoint_flage_ = true;
            ret                           = llm_channel->subscriber_work_id(
                data,
                std::bind(&llm_whisper::vad_endpoint, this, std::weak_ptr<llm_task>(llm_task_obj),
                                                    std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1, std::placeholders::_2));
        }
        if (ret) {
            error_body["code"]    = -20;
            error_body["message"] = "link false";
            send("None", "None", error_body, work_id);
            return;
        } else {
            send("None", "None", LLM_NO_ERROR, work_id);
        }
    }

    void unlink(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_whisper::unlink:%s", data.c_str());
        int ret = 0;
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return;
        }
        auto llm_channel  = get_channel(work_id);
        auto llm_task_obj = llm_task_[work_id_num];
        if (llm_task_obj->superior_id_ == work_id) {
            llm_task_obj->superior_flage_ = false;
        }
        llm_channel->stop_subscriber_work_id(data);
        for (auto it = llm_task_obj->inputs_.begin(); it != llm_task_obj->inputs_.end();) {
            if (*it == data) {
                it = llm_task_obj->inputs_.erase(it);
            } else {
                ++it;
            }
        }
        send("None", "None", LLM_NO_ERROR, work_id);
    }

    void taskinfo(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_whisper::taskinfo:%s", data.c_str());

        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("whisper.tasklist", req_body, LLM_NO_ERROR, work_id);
        } else {
            if (llm_task_.find(work_id_num) == llm_task_.end()) {
                req_body["code"]    = -6;
                req_body["message"] = "Unit Does Not Exist";
                send("None", "None", req_body, work_id);
                return;
            }
            auto llm_task_obj           = llm_task_[work_id_num];
            req_body["model"]           = llm_task_obj->model_;
            req_body["response_format"] = llm_task_obj->response_format_;
            req_body["enoutput"]        = llm_task_obj->enoutput_;
            req_body["inputs"]          = llm_task_obj->inputs_;
            send("whisper.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        SLOGI("llm_whisper::exit:%s", data.c_str());
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return -1;
        }
        llm_task_[work_id_num]->stop();
        auto llm_channel = get_channel(work_id_num);
        llm_channel->stop_subscriber("");
        if (llm_task_[work_id_num]->audio_flage_) {
            unit_call("audio", "cap_stop", "None");
        }
        llm_task_.erase(work_id_num);
        send("None", "None", LLM_NO_ERROR, work_id);
        return 0;
    }

    ~llm_whisper()
    {
        while (1) {
            auto iteam = llm_task_.begin();
            if (iteam == llm_task_.end()) {
                break;
            }
            iteam->second->stop();
            if (iteam->second->audio_flage_) {
                unit_call("audio", "cap_stop", "None");
            }
            get_channel(iteam->first)->stop_subscriber("");
            iteam->second.reset();
            llm_task_.erase(iteam->first);
        }
    }
};

int main(int argc, char *argv[])
{
    signal(SIGTERM, __sigint);
    signal(SIGINT, __sigint);
    mkdir("/tmp/llm", 0777);
    llm_whisper llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}