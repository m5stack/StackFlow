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
#include "../../../../SDK/components/utilities/include/sample_log.h"

int main_exit_flage = 0;
static void __sigint(int iSigNo)
{
    SLOGW("llm_sys will be exit!");
    main_exit_flage = 1;
}

#include "sample_audio.h"
#include "alsa_audio.h"

#define CONFIG_AUTO_SET(obj, key)             \
    if (config_body.contains(#key))           \
        mode_config_.key = config_body[#key]; \
    else if (obj.contains(#key))              \
        mode_config_.key = obj[#key];

using namespace StackFlows;

class llm_audio : public StackFlow {
private:
    enum { EVENT_LOAD_CONFIG = EVENT_EXPORT + 1, EVENT_QUEUE_PLAY };
    std::atomic_bool audio_clear_flage_;
    std::string sys_pcm_cap_channel = "ipc:///tmp/llm/pcm.cap.socket";
    static llm_audio *self;
    std::unique_ptr<pzmq> pub_ctx_;
    std::atomic_int cap_status_;
    std::unique_ptr<std::thread> audio_play_thread_;
    std::unique_ptr<std::thread> audio_cap_thread_;
    std::mutex ax_play_mtx;

    static void on_cap_sample(const char *data, int size)
    {
        self->pub_ctx_->send_data((const char *)data, size);
    }

    void hw_queue_play(const std::shared_ptr<void> &arg)
    {
        if (audio_clear_flage_) {
            return;
        }
        std::shared_ptr<pzmq_data> originalPtr = std::static_pointer_cast<pzmq_data>(arg);
        std::lock_guard<std::mutex> guard(ax_play_mtx);
        ax_play(play_config.card, play_config.device, play_config.volume, play_config.channel, play_config.rate,
                play_config.bit, originalPtr->data(), originalPtr->size());
    }

    void hw_play(const std::string &audio_data)
    {
        std::lock_guard<std::mutex> guard(ax_play_mtx);
        ax_play(play_config.card, play_config.device, play_config.volume, play_config.channel, play_config.rate,
                play_config.bit, audio_data.c_str(), audio_data.length());
    }

    void hw_cap()
    {
#if (defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)) && !defined(CONFIG_AXCL_ENABLED)
        ax_cap_start(cap_config.card, cap_config.device, cap_config.volume, cap_config.channel, cap_config.rate,
                     cap_config.bit, llm_audio::on_cap_sample);
#else
        alsa_cap_start(cap_config.card, cap_config.device, cap_config.volume, cap_config.channel, cap_config.rate,
                       cap_config.bit, llm_audio::on_cap_sample);
#endif
    }

    void _play(const std::string &audio_data)
    {
        if (audio_play_thread_) {
            _play_stop();
        }
        audio_play_thread_ = std::make_unique<std::thread>(std::bind(&llm_audio::hw_play, this, audio_data));
    }

    void _play_stop()
    {
        ax_close_play();
        if (audio_play_thread_) {
            audio_play_thread_->join();
            audio_play_thread_.reset();
        }
    }

    void _cap()
    {
        if (!audio_cap_thread_) {
            pub_ctx_          = std::make_unique<pzmq>(sys_pcm_cap_channel, ZMQ_PUB);
            audio_cap_thread_ = std::make_unique<std::thread>(std::bind(&llm_audio::hw_cap, this));
        }
    }

    void _cap_stop()
    {
#if (defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)) && !defined(CONFIG_AXCL_ENABLED)
        ax_close_cap();

#else
        alsa_close_cap();
#endif
        if (audio_cap_thread_) {
            audio_cap_thread_->join();
            audio_cap_thread_.reset();
            pub_ctx_.reset();
        }
    }

public:
    llm_audio() : StackFlow("audio")
    {
        event_queue_.appendListener(EVENT_QUEUE_PLAY,
                                    std::bind(&llm_audio::hw_queue_play, this, std::placeholders::_1));
        setup("", "audio.play", "{\"None\":\"None\"}");
        setup("", "audio.cap", "{\"None\":\"None\"}");
        self        = this;
        cap_status_ = 0;
        rpc_ctx_->register_rpc_action("play",
                                      std::bind(&llm_audio::play, this, std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action(
            "play_raw", std::bind(&llm_audio::play_raw, this, std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action(
            "queue_play", std::bind(&llm_audio::enqueue_play, this, std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action(
            "play_stop", std::bind(&llm_audio::play_stop, this, std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action("queue_play_stop", std::bind(&llm_audio::queue_play_stop, this,
                                                                   std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action(
            "audio_status", std::bind(&llm_audio::audio_status, this, std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action("cap",
                                      std::bind(&llm_audio::cap, this, std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action(
            "cap_stop", std::bind(&llm_audio::cap_stop, this, std::placeholders::_1, std::placeholders::_2));
        rpc_ctx_->register_rpc_action(
            "cap_stop_all", std::bind(&llm_audio::cap_stop_all, this, std::placeholders::_1, std::placeholders::_2));
    }
    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json config_body;
        nlohmann::json file_body;
        nlohmann::json error_body;
        std::string base_model_path;
        std::string base_model_config_path;
#if (defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)) && !defined(CONFIG_AXCL_ENABLED)
        std::list<std::string> config_file_paths;
        if (access("/sys/devices/platform/soc/4851000.i2c/i2c-1/1-0043", F_OK) == 0) {
            config_file_paths = get_config_file_paths(base_model_path, base_model_config_path, "audio_kit");
        } else {
            config_file_paths = get_config_file_paths(base_model_path, base_model_config_path, "audio");
        }
#else
        std::list<std::string> config_file_paths =
            get_config_file_paths(base_model_path, base_model_config_path, "audio");
#endif
        try {
            config_body = nlohmann::json::parse(data);
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
                error_body["code"]    = -22;
                error_body["message"] = "all config file miss.";
                send("None", "None", error_body, "audio");
                SLOGE("all config file miss");
                return -2;
            }
        } catch (...) {
            error_body["code"]    = -22;
            error_body["message"] = "config file parameter format error.";
            send("None", "None", error_body, "audio");
            return -2;
        }
        AX_AUDIO_SAMPLE_CONFIG_t mode_config_;
        try {
            memset(&mode_config_, 0, sizeof(AX_AUDIO_SAMPLE_CONFIG_t));
            if (object == "audio.play") {
                CONFIG_AUTO_SET(file_body["play_param"], stPoolConfig.MetaSize);
                CONFIG_AUTO_SET(file_body["play_param"], stPoolConfig.BlkSize);
                CONFIG_AUTO_SET(file_body["play_param"], stPoolConfig.BlkCnt);
                CONFIG_AUTO_SET(file_body["play_param"], stPoolConfig.IsMergeMode);
                CONFIG_AUTO_SET(file_body["play_param"], stPoolConfig.CacheMode);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.enBitwidth);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.enSoundmode);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.u32ChnCnt);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.enLinkMode);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.enSamplerate);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.U32Depth);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.u32PeriodSize);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.u32PeriodCount);
                CONFIG_AUTO_SET(file_body["play_param"], stAttr.bInsertSilence);
                CONFIG_AUTO_SET(file_body["play_param"], stVqeAttr.s32SampleRate);
                CONFIG_AUTO_SET(file_body["play_param"], stVqeAttr.stNsCfg.bNsEnable);
                CONFIG_AUTO_SET(file_body["play_param"], stVqeAttr.stNsCfg.enAggressivenessLevel);
                CONFIG_AUTO_SET(file_body["play_param"], stVqeAttr.stAgcCfg.bAgcEnable);
                CONFIG_AUTO_SET(file_body["play_param"], stVqeAttr.stAgcCfg.enAgcMode);
                CONFIG_AUTO_SET(file_body["play_param"], stVqeAttr.stAgcCfg.s16TargetLevel);
                CONFIG_AUTO_SET(file_body["play_param"], stVqeAttr.stAgcCfg.s16Gain);
#if (defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)) && !defined(CONFIG_AXCL_ENABLED)
                CONFIG_AUTO_SET(file_body["play_param"], stHpfAttr.bEnable);
                CONFIG_AUTO_SET(file_body["play_param"], stHpfAttr.s32GainDb);
                CONFIG_AUTO_SET(file_body["play_param"], stHpfAttr.s32Freq);
                CONFIG_AUTO_SET(file_body["play_param"], stLpfAttr.bEnable);
                CONFIG_AUTO_SET(file_body["play_param"], stLpfAttr.s32GainDb);
                CONFIG_AUTO_SET(file_body["play_param"], stLpfAttr.s32Samplerate);
                CONFIG_AUTO_SET(file_body["play_param"], stLpfAttr.s32Freq);
                CONFIG_AUTO_SET(file_body["play_param"], stEqAttr.bEnable);
                CONFIG_AUTO_SET(file_body["play_param"], stEqAttr.s32GainDb[0]);
                CONFIG_AUTO_SET(file_body["play_param"], stEqAttr.s32GainDb[1]);
                CONFIG_AUTO_SET(file_body["play_param"], stEqAttr.s32GainDb[2]);
                CONFIG_AUTO_SET(file_body["play_param"], stEqAttr.s32GainDb[3]);
                CONFIG_AUTO_SET(file_body["play_param"], stEqAttr.s32GainDb[4]);
                CONFIG_AUTO_SET(file_body["play_param"], stEqAttr.s32Samplerate);
#endif
                CONFIG_AUTO_SET(file_body["play_param"], gResample);
                CONFIG_AUTO_SET(file_body["play_param"], enInSampleRate);
                CONFIG_AUTO_SET(file_body["play_param"], gInstant);
                CONFIG_AUTO_SET(file_body["play_param"], gInsertSilence);
                CONFIG_AUTO_SET(file_body["play_param"], card);
                CONFIG_AUTO_SET(file_body["play_param"], device);
                CONFIG_AUTO_SET(file_body["play_param"], volume);
                CONFIG_AUTO_SET(file_body["play_param"], channel);
                CONFIG_AUTO_SET(file_body["play_param"], rate);
                CONFIG_AUTO_SET(file_body["play_param"], bit);
                if (config_body.contains("stPoolConfig.PartitionName")) {
                    std::string PartitionName = config_body["stPoolConfig.PartitionName"];
                    for (int i = 0; i < PartitionName.length(); i++) {
                        mode_config_.stPoolConfig.PartitionName[i] = PartitionName[i];
                    }
                } else if (file_body["cap_param"].contains("stPoolConfig.PartitionName")) {
                    std::string PartitionName = file_body["cap_param"]["stPoolConfig.PartitionName"];
                    for (int i = 0; i < PartitionName.length(); i++) {
                        mode_config_.stPoolConfig.PartitionName[i] = PartitionName[i];
                    }
                }
                memcpy(&play_config, &mode_config_, sizeof(AX_AUDIO_SAMPLE_CONFIG_t));
            }
            if (object == "audio.cap") {
                CONFIG_AUTO_SET(file_body["cap_param"], stPoolConfig.MetaSize);
                CONFIG_AUTO_SET(file_body["cap_param"], stPoolConfig.BlkSize);
                CONFIG_AUTO_SET(file_body["cap_param"], stPoolConfig.BlkCnt);
                CONFIG_AUTO_SET(file_body["cap_param"], stPoolConfig.IsMergeMode);
                CONFIG_AUTO_SET(file_body["cap_param"], stPoolConfig.CacheMode);
                CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.enBitwidth);
                CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.enLinkMode);
                CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.enSamplerate);
                // CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.enLayoutMode);
                if (config_body.contains("aistAttr.enLayoutMode"))
                    mode_config_.aistAttr.enLayoutMode = config_body["aistAttr.enLayoutMode"];
                else if (file_body["cap_param"].contains("aistAttr.enLayoutMode")) {
                    mode_config_.aistAttr.enLayoutMode = file_body["cap_param"]["aistAttr.enLayoutMode"];
                    if (access("/sys/devices/platform/soc/4851000.i2c/i2c-1/1-0043", F_OK) == 0) {
                        if (mode_config_.aistAttr.enLayoutMode == AX_AI_REF_MIC) {
                            mode_config_.aistAttr.enLayoutMode = AX_AI_MIC_REF;
                        }
                    }
                }
                CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.U32Depth);
                CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.u32PeriodSize);
                CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.u32PeriodCount);
                CONFIG_AUTO_SET(file_body["cap_param"], aistAttr.u32ChnCnt);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.s32SampleRate);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.u32FrameSamples);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.stNsCfg.bNsEnable);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.stNsCfg.enAggressivenessLevel);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.stAgcCfg.bAgcEnable);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.stAgcCfg.enAgcMode);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.stAgcCfg.s16TargetLevel);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.stAgcCfg.s16Gain);
                CONFIG_AUTO_SET(file_body["cap_param"], aistVqeAttr.stAecCfg.enAecMode);
#if (defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)) && !defined(CONFIG_AXCL_ENABLED)
                CONFIG_AUTO_SET(file_body["cap_param"], stHpfAttr.bEnable);
                CONFIG_AUTO_SET(file_body["cap_param"], stHpfAttr.s32GainDb);
                CONFIG_AUTO_SET(file_body["cap_param"], stHpfAttr.s32Samplerate);
                CONFIG_AUTO_SET(file_body["cap_param"], stHpfAttr.s32Freq);
                CONFIG_AUTO_SET(file_body["cap_param"], stLpfAttr.bEnable);
                CONFIG_AUTO_SET(file_body["cap_param"], stLpfAttr.s32GainDb);
                CONFIG_AUTO_SET(file_body["cap_param"], stLpfAttr.s32Samplerate);
                CONFIG_AUTO_SET(file_body["cap_param"], stLpfAttr.s32Freq);
                CONFIG_AUTO_SET(file_body["cap_param"], stEqAttr.bEnable);
                CONFIG_AUTO_SET(file_body["cap_param"], stEqAttr.s32GainDb[0]);
                CONFIG_AUTO_SET(file_body["cap_param"], stEqAttr.s32GainDb[1]);
                CONFIG_AUTO_SET(file_body["cap_param"], stEqAttr.s32GainDb[2]);
                CONFIG_AUTO_SET(file_body["cap_param"], stEqAttr.s32GainDb[3]);
                CONFIG_AUTO_SET(file_body["cap_param"], stEqAttr.s32GainDb[4]);
                CONFIG_AUTO_SET(file_body["cap_param"], stEqAttr.s32Samplerate);
#endif
                CONFIG_AUTO_SET(file_body["cap_param"], gResample);
                CONFIG_AUTO_SET(file_body["cap_param"], enOutSampleRate);
                CONFIG_AUTO_SET(file_body["cap_param"], gDbDetection);
                CONFIG_AUTO_SET(file_body["cap_param"], card);
                CONFIG_AUTO_SET(file_body["cap_param"], device);
                CONFIG_AUTO_SET(file_body["cap_param"], volume);
                CONFIG_AUTO_SET(file_body["cap_param"], channel);
                CONFIG_AUTO_SET(file_body["cap_param"], rate);
                CONFIG_AUTO_SET(file_body["cap_param"], bit);

                if (config_body.contains("stPoolConfig.PartitionName")) {
                    std::string PartitionName = config_body["stPoolConfig.PartitionName"];
                    for (int i = 0; i < PartitionName.length(); i++) {
                        mode_config_.stPoolConfig.PartitionName[i] = PartitionName[i];
                    }
                } else if (file_body["cap_param"].contains("stPoolConfig.PartitionName")) {
                    std::string PartitionName = file_body["cap_param"]["stPoolConfig.PartitionName"];
                    for (int i = 0; i < PartitionName.length(); i++) {
                        mode_config_.stPoolConfig.PartitionName[i] = PartitionName[i];
                    }
                }
                if (config_body.contains("sys_pcm_cap_channel")) {
                    sys_pcm_cap_channel = config_body["sys_pcm_cap_channel"];
                } else if (file_body["cap_param"].contains("sys_pcm_cap_channel")) {
                    sys_pcm_cap_channel = file_body["cap_param"]["sys_pcm_cap_channel"];
                }
                memcpy(&cap_config, &mode_config_, sizeof(AX_AUDIO_SAMPLE_CONFIG_t));
            }
        } catch (...) {
            error_body["code"]    = -22;
            error_body["message"] = "Parameter format error.";
            send("None", "None", error_body, "audio");
            return -3;
        }
        send(std::string("None"), std::string("None"), std::string(""), "audio");
        return -1;
    }

    bool destream(const std::string &indata, std::string &outdata)
    {
        static std::unordered_map<int, std::string> _stream_data_buff;
        int index                = std::stoi(sample_json_str_get(indata, "index"));
        std::string finish       = sample_json_str_get(indata, "finish");
        std::string delta        = sample_json_str_get(indata, "delta");
        _stream_data_buff[index] = delta;
        try {
            if (finish == "true") {
                for (size_t i = 0; i < _stream_data_buff.size(); i++) {
                    outdata += _stream_data_buff.at(i);
                }
                _stream_data_buff.clear();
                return false;
            }
        } catch (...) {
            _stream_data_buff.clear();
        }
        return true;
    }

    std::string dewav(const std::string &rawdata)
    {
        int post = 0;
        if (rawdata.length() > 10)
            for (int i = 0; i < rawdata.length() - 4; i++) {
                if ((rawdata[i] == 'd') && (rawdata[i + 1] == 'a') && (rawdata[i + 2] == 't') &&
                    (rawdata[i + 3] == 'a')) {
                    post = i + 8;
                    break;
                }
            }
        if (post > 0) {
            _play(std::string((char *)(rawdata.c_str() + post), rawdata.length() - post));
        } else {
            return std::string("wav_error");
        }
        return std::string("playwav");
    }

    std::string demp3(const std::string &rawdata)
    {
        return LLM_NONE;
    }

    std::string depcm(const std::string &rawdata)
    {
        return LLM_NONE;
    }

    std::string parse_data(const std::string &object, const std::string &data)
    {
        std::string _data;
        bool s   = (object.find("stream") == std::string::npos) ? false : true;
        bool wav = (object.find("wav") == std::string::npos) ? false : true;
        bool mp3 = (object.find("mp3") == std::string::npos) ? false : true;
        bool pcm = (object.find("pcm") == std::string::npos) ? false : true;
        if (s) {
            if (destream(data, _data)) return std::string("destream");
        } else {
            _data = data;
        }
        std::vector<unsigned char> wav_buff(BASE64_DECODE_OUT_SIZE(_data.length()));
        int ret = base64_decode(_data.c_str(), _data.length(), wav_buff.data());
        if (ret <= 0) return std::string("base64_error");
        if (wav) return dewav(std::string((char *)wav_buff.data(), ret));
        if (mp3) return demp3(std::string((char *)wav_buff.data(), ret));
        if (pcm) return depcm(std::string((char *)wav_buff.data(), ret));
        return LLM_NONE;
    }

    std::string play(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        std::string zmq_url    = rawdata->get_param(0);
        std::string audio_json = rawdata->get_param(1);
        std::string ret_val =
            parse_data(sample_json_str_get(audio_json, "object"), sample_json_str_get(audio_json, "data"));
        request_id_ = sample_json_str_get(audio_json, "request_id");
        send(LLM_NONE, LLM_NONE, LLM_NO_ERROR, sample_json_str_get(audio_json, "work_id"), zmq_url);
        return ret_val;
    }

    std::string play_raw(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        auto _rawdata = rawdata->string();
        if (_rawdata.empty()) return std::string("rawdata empty");
        _play(_rawdata);
        return LLM_NONE;
    }

    std::string enqueue_play(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        audio_clear_flage_ = false;
        event_queue_.enqueue(EVENT_QUEUE_PLAY, rawdata);
        return LLM_NONE;
    }

    std::string audio_status(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        std::string _rawdata = rawdata->string();
        if (_rawdata == "play") {
            if (ax_play_status()) {
                return std::string("None");
            } else {
                return std::string("Runing");
            }
        } else if (_rawdata == "cap") {
            if (ax_cap_status()) {
                return std::string("None");
            } else {
                return std::string("Runing");
            }
        } else {
            std::ostringstream return_val;
            return_val << "{\"play\":";
            if (ax_play_status()) {
                return_val << "\"None\"";
            } else {
                return_val << "\"Runing\"";
            }
            return_val << "\"cap\":";
            if (ax_cap_status()) {
                return_val << "\"None\"";
            } else {
                return_val << "\"Runing\"";
            }
            return_val << "\"}";
            return return_val.str();
        }
    }

    std::string play_stop(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        _play_stop();
        return LLM_NONE;
    }

    std::string queue_play_stop(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        audio_clear_flage_ = true;
        return LLM_NONE;
    }

    std::string cap(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        if (cap_status_ == 0) {
            _cap();
        }
        cap_status_++;
        return sys_pcm_cap_channel;
    }

    std::string cap_stop(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        if (cap_status_ > 0) {
            cap_status_--;
            if (cap_status_ == 0) {
                _cap_stop();
            }
        }
        return LLM_NONE;
    }

    std::string cap_stop_all(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        cap_status_ = 0;
        _cap_stop();
        return LLM_NONE;
    }

    ~llm_audio()
    {
        _play_stop();
        _cap_stop();
    }
};

llm_audio *llm_audio::self;
int main(int argc, char *argv[])
{
    signal(SIGTERM, __sigint);
    signal(SIGINT, __sigint);
    mkdir("/tmp/llm", 0777);
    llm_audio audio;
    while (!main_exit_flage) {
        sleep(1);
    }
    audio.llm_firework_exit();
    return 0;
}