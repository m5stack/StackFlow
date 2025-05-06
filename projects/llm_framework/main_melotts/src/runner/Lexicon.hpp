#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <iostream>
#include "../../../../../SDK/components/utilities/include/sample_log.h"

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}
class Lexicon {
private:
    std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<int>>> lexicon;
    size_t max_phrase_length;
    std::pair<std::vector<int>, std::vector<int>> unknown_token;
    std::unordered_map<int, std::string> reverse_tokens;

public:
    Lexicon(const std::string& lexicon_filename, const std::string& tokens_filename) : max_phrase_length(0)
    {
        SLOGI("词典加载: %zu 发音表加载: %zu", tokens_filename, lexicon_filename);
        std::unordered_map<std::string, int> tokens;
        std::ifstream ifs(tokens_filename);
        assert(ifs.is_open());
        std::string line;
        while (std::getline(ifs, line)) {
            auto splitted_line = split(line, ' ');
            if (splitted_line.size() >= 2) {
                int token_id = std::stoi(splitted_line[1]);
                tokens.insert({splitted_line[0], token_id});
                reverse_tokens[token_id] = splitted_line[0];
            }
        }
        ifs.close();
        ifs.open(lexicon_filename);
        assert(ifs.is_open());
        while (std::getline(ifs, line)) {
            auto splitted_line = split(line, ' ');
            if (splitted_line.empty()) continue;
            std::string word_or_phrase = splitted_line[0];
            auto chars                 = splitEachChar(word_or_phrase);
            max_phrase_length          = std::max(max_phrase_length, chars.size());
            size_t phone_tone_len      = splitted_line.size() - 1;
            size_t half_len            = phone_tone_len / 2;
            std::vector<int> phones, tones;
            for (size_t i = 0; i < phone_tone_len; i++) {
                auto phone_or_tone = splitted_line[i + 1];
                if (i < half_len) {
                    if (tokens.find(phone_or_tone) != tokens.end()) {
                        phones.push_back(tokens[phone_or_tone]);
                    }
                } else {
                    tones.push_back(std::stoi(phone_or_tone));
                }
            }
            lexicon[word_or_phrase] = std::make_pair(phones, tones);
        }
        const std::vector<std::string> punctuation{"!", "?", "…", ",", ".", "'", "-"};
        for (const auto& p : punctuation) {
            if (tokens.find(p) != tokens.end()) {
                int i      = tokens[p];
                lexicon[p] = std::make_pair(std::vector<int>{i}, std::vector<int>{0});
            }
        }
        assert(tokens.find("_") != tokens.end());
        unknown_token = std::make_pair(std::vector<int>{tokens["_"]}, std::vector<int>{0});
        lexicon[" "]  = unknown_token;
        lexicon["，"] = lexicon[","];
        lexicon["。"] = lexicon["."];
        lexicon["！"] = lexicon["!"];
        lexicon["？"] = lexicon["?"];
        SLOGI("词典加载完成，包含 %zu 个条目，最长词组长度: %zu", lexicon.size(), max_phrase_length);
    }
    std::vector<std::string> splitEachChar(const std::string& text)
    {
        std::vector<std::string> words;
        int len = text.length();
        int i   = 0;
        while (i < len) {
            int next = 1;
            if ((text[i] & 0x80) == 0x00) {
                // ASCII
            } else if ((text[i] & 0xE0) == 0xC0) {
                next = 2;  // 2字节UTF-8
            } else if ((text[i] & 0xF0) == 0xE0) {
                next = 3;  // 3字节UTF-8
            } else if ((text[i] & 0xF8) == 0xF0) {
                next = 4;  // 4字节UTF-8
            }
            words.push_back(text.substr(i, next));
            i += next;
        }
        return words;
    }
    bool is_english(const std::string& s)
    {
        return s.size() == 1 && ((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z'));
    }

    bool is_english_token_char(const std::string& s)
    {
        if (s.size() != 1) return false;
        char c = s[0];
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
    }

    void process_unknown_english(const std::string& word, std::vector<int>& phones, std::vector<int>& tones)
    {
        SLOGI("Processing unknown term: %s", word.c_str());

        std::string orig_word = word;
        std::vector<std::string> parts;
        std::vector<std::string> phonetic_parts;

        size_t start = 0;
        while (start < word.size()) {
            bool matched = false;

            for (size_t len = std::min(word.size() - start, (size_t)10); len > 0 && !matched; --len) {
                std::string sub_word       = word.substr(start, len);
                std::string lower_sub_word = sub_word;
                std::transform(lower_sub_word.begin(), lower_sub_word.end(), lower_sub_word.begin(),
                               [](unsigned char c) { return std::tolower(c); });

                if (lexicon.find(lower_sub_word) != lexicon.end()) {
                    // Substring found in lexicon
                    auto& [sub_phones, sub_tones] = lexicon[lower_sub_word];
                    phones.insert(phones.end(), sub_phones.begin(), sub_phones.end());
                    tones.insert(tones.end(), sub_tones.begin(), sub_tones.end());

                    parts.push_back(sub_word);
                    phonetic_parts.push_back(phonesToString(sub_phones));

                    SLOGI("  Matched: '%s' -> %s", sub_word.c_str(), phonesToString(sub_phones).c_str());

                    start += len;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                std::string single_char = word.substr(start, 1);
                std::string lower_char  = single_char;
                std::transform(lower_char.begin(), lower_char.end(), lower_char.begin(),
                               [](unsigned char c) { return std::tolower(c); });

                if (lexicon.find(lower_char) != lexicon.end()) {
                    auto& [char_phones, char_tones] = lexicon[lower_char];
                    phones.insert(phones.end(), char_phones.begin(), char_phones.end());
                    tones.insert(tones.end(), char_tones.begin(), char_tones.end());

                    parts.push_back(single_char);
                    phonetic_parts.push_back(phonesToString(char_phones));

                    SLOGI("  Single char: '%s' -> %s", single_char.c_str(), phonesToString(char_phones).c_str());
                } else {
                    phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
                    tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());

                    parts.push_back(single_char);
                    phonetic_parts.push_back("_unknown_");

                    SLOGI("  Unknown: '%s'", single_char.c_str());
                }

                start++;
            }
        }

        std::string parts_str, phonetic_str;
        for (size_t i = 0; i < parts.size(); i++) {
            if (i > 0) {
                parts_str += " ";
                phonetic_str += " ";
            }
            parts_str += parts[i];
            phonetic_str += phonetic_parts[i];
        }

        SLOGI("%s\t|\tDecomposed: %s\t|\tPhonetics: %s", orig_word.c_str(), parts_str.c_str(), phonetic_str.c_str());
    }
    void convert(const std::string& text, std::vector<int>& phones, std::vector<int>& tones)
    {
        SLOGI("\n开始处理文本: \"%s\"", text.c_str());
        SLOGI("=======匹配结果=======");
        SLOGI("单元\t|\t音素\t|\t声调");
        SLOGI("-----------------------------");
        phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
        tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());

        SLOGI("<BOS>\t|\t%s\t|\t%s", phonesToString(unknown_token.first).c_str(),
              tonesToString(unknown_token.second).c_str());
        auto chars = splitEachChar(text);
        int i      = 0;
        while (i < chars.size()) {
            if (is_english(chars[i])) {
                std::string eng_word;
                int start = i;
                while (i < chars.size() && is_english(chars[i])) {
                    eng_word += chars[i++];
                }
                std::string orig_word = eng_word;
                std::transform(eng_word.begin(), eng_word.end(), eng_word.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (lexicon.find(eng_word) != lexicon.end()) {
                    auto& [eng_phones, eng_tones] = lexicon[eng_word];
                    phones.insert(phones.end(), eng_phones.begin(), eng_phones.end());
                    tones.insert(tones.end(), eng_tones.begin(), eng_tones.end());
                    SLOGI("%s\t|\t%s\t|\t%s", orig_word.c_str(), phonesToString(eng_phones).c_str(),
                          tonesToString(eng_tones).c_str());
                } else {
                    process_unknown_english(orig_word, phones, tones);
                }
                continue;
            }
            std::string c = chars[i++];
            if (c == " ") continue;
            i--;
            bool matched = false;
            for (size_t len = std::min(max_phrase_length, chars.size() - i); len > 0 && !matched; --len) {
                std::string phrase;
                for (size_t j = 0; j < len; ++j) {
                    phrase += chars[i + j];
                }
                if (lexicon.find(phrase) != lexicon.end()) {
                    auto& [phrase_phones, phrase_tones] = lexicon[phrase];
                    phones.insert(phones.end(), phrase_phones.begin(), phrase_phones.end());
                    tones.insert(tones.end(), phrase_tones.begin(), phrase_tones.end());
                    SLOGI("%s\t|\t%s\t|\t%s", phrase.c_str(), phonesToString(phrase_phones).c_str(),
                          tonesToString(phrase_tones).c_str());
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                std::string c         = chars[i++];
                std::string s         = c;
                std::string orig_char = s;
                if (s == "，")
                    s = ",";
                else if (s == "。")
                    s = ".";
                else if (s == "！")
                    s = "!";
                else if (s == "？")
                    s = "?";
                if (lexicon.find(s) != lexicon.end()) {
                    auto& [char_phones, char_tones] = lexicon[s];
                    phones.insert(phones.end(), char_phones.begin(), char_phones.end());
                    tones.insert(tones.end(), char_tones.begin(), char_tones.end());
                    SLOGI("%s\t|\t%s\t|\t%s", orig_char.c_str(), phonesToString(char_phones).c_str(),
                          tonesToString(char_tones).c_str());
                } else {
                    phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
                    tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());
                    SLOGI("%s\t|\t%s (未匹配)\t|\t%s", orig_char.c_str(), phonesToString(unknown_token.first).c_str(),
                          tonesToString(unknown_token.second).c_str());
                }
            }
        }
        phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
        tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());
        SLOGI("<EOS>\t|\t%s\t|\t%s", phonesToString(unknown_token.first).c_str(),
              tonesToString(unknown_token.second).c_str());
        SLOGI("\n处理结果汇总:");
        SLOGI("原文: %s", text.c_str());
        SLOGI("音素: %s", phonesToString(phones).c_str());
        SLOGI("声调: %s", tonesToString(tones).c_str());
        SLOGI("====================");
    }

private:
    void processChar(const std::string& c, std::vector<int>& phones, std::vector<int>& tones)
    {
        std::string s = c;
        if (s == "，")
            s = ",";
        else if (s == "。")
            s = ".";
        else if (s == "！")
            s = "!";
        else if (s == "？")
            s = "?";
        auto& phones_and_tones = (lexicon.find(s) != lexicon.end()) ? lexicon[s] : unknown_token;
        phones.insert(phones.end(), phones_and_tones.first.begin(), phones_and_tones.first.end());
        tones.insert(tones.end(), phones_and_tones.second.begin(), phones_and_tones.second.end());
    }
    std::string phonesToString(const std::vector<int>& phones)
    {
        std::string result;
        for (auto id : phones) {
            if (!result.empty()) result += " ";
            if (reverse_tokens.find(id) != reverse_tokens.end()) {
                result += reverse_tokens[id];
            } else {
                result += "<" + std::to_string(id) + ">";
            }
        }
        return result;
    }
    std::string tonesToString(const std::vector<int>& tones)
    {
        std::string result;
        for (auto tone : tones) {
            if (!result.empty()) result += " ";
            result += std::to_string(tone);
        }
        return result;
    }
};