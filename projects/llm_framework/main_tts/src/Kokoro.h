/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <string>
#include <vector>
#include <memory>

namespace kokoro {
struct SentenceInfo {
    std::string sentence;
    std::vector<int> input_ids;
    std::string phonemes;
    int content_len = 0;
    bool is_long    = false;
    std::vector<SentenceInfo> sub_results;

    SentenceInfo() = default;
    SentenceInfo(const std::string &s, const std::vector<int> &ids, const std::string &ph, int len)
        : sentence(s), input_ids(ids), phonemes(ph), content_len(len), is_long(false)
    {
    }

    SentenceInfo(const std::string &s, const std::vector<SentenceInfo> &subs)
        : sentence(s), is_long(true), sub_results(subs)
    {
    }
};

struct MergedGroup {
    bool is_long_split = false;
    std::vector<int> input_ids;
    std::string phonemes;
    std::vector<SentenceInfo> sub_results;

    MergedGroup() = default;
    MergedGroup(const std::vector<int> &ids, const std::string &ph) : is_long_split(false), input_ids(ids), phonemes(ph)
    {
    }

    MergedGroup(const std::vector<SentenceInfo> &subs) : is_long_split(true), sub_results(subs)
    {
    }
};

class Kokoro {
public:
    Kokoro();
    ~Kokoro();
    Kokoro(const Kokoro &)            = delete;
    Kokoro &operator=(const Kokoro &) = delete;

    bool init(const std::string &model_path, int max_seq_len = 96, const std::string &lang_code = "z",
              const std::string &voices_path = "./voices", const std::string &voice_name = "af_heart",
              const std::string &vocab_path       = "dict/vocab.txt",
              const std::string &espeak_data_path = "./espeak-ng-data", const std::string &dict_dir = "./dict");

    bool tts(const std::string &text, const std::string &voice_name, float speed, int sample_rate, float fade_out,
             float pause_duration, std::vector<float> &generated_audio);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kokoro