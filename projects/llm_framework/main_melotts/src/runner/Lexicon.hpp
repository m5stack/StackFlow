#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <iostream> // 用于日志输出

// 使用引用传参优化split函数，避免不必要的拷贝
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        if (!item.empty()) { // 避免添加空字符串
            result.push_back(item);
        }
    }
    return result;
}

class Lexicon {
private:
    std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<int>>> lexicon;
    size_t max_phrase_length; // 追踪词典中最长的词组长度
    std::pair<std::vector<int>, std::vector<int>> unknown_token; // '_'的发音作为未知词的默认值
    std::unordered_map<int, std::string> reverse_tokens; // 用于将音素ID转回音素符号，用于日志

public:
    Lexicon(const std::string& lexicon_filename, const std::string& tokens_filename) : max_phrase_length(0) {
        std::unordered_map<std::string, int> tokens;
        
        // 加载tokens
        std::ifstream ifs(tokens_filename);
        assert(ifs.is_open());

        std::string line;
        while (std::getline(ifs, line)) {
            auto splitted_line = split(line, ' ');
            if (splitted_line.size() >= 2) {
                int token_id = std::stoi(splitted_line[1]);
                tokens.insert({splitted_line[0], token_id});
                reverse_tokens[token_id] = splitted_line[0]; // 建立反向映射
            }
        }
        ifs.close();

        // 加载lexicon
        ifs.open(lexicon_filename);
        assert(ifs.is_open());
        while (std::getline(ifs, line)) {
            auto splitted_line = split(line, ' ');
            if (splitted_line.empty()) continue;
            
            std::string word_or_phrase = splitted_line[0];
            
            // 更新最长词组长度
            auto chars = splitEachChar(word_or_phrase);
            max_phrase_length = std::max(max_phrase_length, chars.size());
            
            size_t phone_tone_len = splitted_line.size() - 1;
            size_t half_len = phone_tone_len / 2;
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

        // 添加特殊映射
        lexicon["呣"] = lexicon["母"];
        lexicon["嗯"] = lexicon["恩"];

        // 添加标点符号
        const std::vector<std::string> punctuation{"!", "?", "…", ",", ".", "'", "-"};
        for (const auto& p : punctuation) {
            if (tokens.find(p) != tokens.end()) {
                int i = tokens[p];
                lexicon[p] = std::make_pair(std::vector<int>{i}, std::vector<int>{0});
            }
        }
        
        // 设置'_'作为未知词的发音
        assert(tokens.find("_") != tokens.end());  // 确保tokens中包含"_"
        unknown_token = std::make_pair(std::vector<int>{tokens["_"]}, std::vector<int>{0});
        
        // 空格映射到'_'的发音
        lexicon[" "] = unknown_token;

        // 中文标点转换映射
        lexicon["，"] = lexicon[","];
        lexicon["。"] = lexicon["."];
        lexicon["！"] = lexicon["!"];
        lexicon["？"] = lexicon["?"];
        
        // 输出词典信息
        std::cout << "词典加载完成，包含 " << lexicon.size() << " 个条目，最长词组长度: " << max_phrase_length << std::endl;
    }

    std::vector<std::string> splitEachChar(const std::string& text) {
        std::vector<std::string> words;
        int len = text.length();
        int i = 0;
        
        while (i < len) {
            int next = 1;
            if ((text[i] & 0x80) == 0x00) {
                // ASCII
            } else if ((text[i] & 0xE0) == 0xC0) {
                next = 2; // 2字节UTF-8
            } else if ((text[i] & 0xF0) == 0xE0) {
                next = 3; // 3字节UTF-8
            } else if ((text[i] & 0xF8) == 0xF0) {
                next = 4; // 4字节UTF-8
            }
            words.push_back(text.substr(i, next));
            i += next;
        }
        return words;
    } 

    bool is_english(const std::string& s) {
        return s.size() == 1 && ((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z'));
    }

    // 根据词典中的内容，使用最长匹配算法处理输入文本
    void convert(const std::string& text, std::vector<int>& phones, std::vector<int>& tones) {
        std::cout << "\n开始处理文本: \"" << text << "\"" << std::endl;
        std::cout << "=======匹配结果=======" << std::endl;
        std::cout << "单元\t|\t音素\t|\t声调" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        
        // 在开头添加'_'边界标记
        phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
        tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());
        std::cout << "<BOS>\t|\t" << phonesToString(unknown_token.first) << "\t|\t" 
                 << tonesToString(unknown_token.second) << std::endl;
        
        auto chars = splitEachChar(text);
        int i = 0;
        
        while (i < chars.size()) {
            // 处理英文单词
            if (is_english(chars[i])) {
                std::string eng_word;
                int start = i;
                while (i < chars.size() && is_english(chars[i])) {
                    eng_word += chars[i++];
                }
                
                // 英文转小写
                std::string orig_word = eng_word; // 保留原始单词用于日志
                std::transform(eng_word.begin(), eng_word.end(), eng_word.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                
                // 如果词典中有这个英文单词，使用它；否则使用'_'的发音
                if (lexicon.find(eng_word) != lexicon.end()) {
                    auto& [eng_phones, eng_tones] = lexicon[eng_word];
                    phones.insert(phones.end(), eng_phones.begin(), eng_phones.end());
                    tones.insert(tones.end(), eng_tones.begin(), eng_tones.end());
                    
                    // 打印匹配信息
                    std::cout << orig_word << "\t|\t" << phonesToString(eng_phones) << "\t|\t" 
                             << tonesToString(eng_tones) << std::endl;
                } else {
                    // 未找到单词，使用'_'的发音
                    phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
                    tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());
                    
                    // 打印未匹配信息
                    std::cout << orig_word << "\t|\t" << phonesToString(unknown_token.first) << " (未匹配)\t|\t" 
                             << tonesToString(unknown_token.second) << std::endl;
                }
                continue;
            }
            // 处理非英文字符（如空格、标点）
            std::string c = chars[i++];
            if (c == " ") continue; // 跳过空格
            // 回退一步，用于最长匹配
            i--;

            
            // 最长匹配算法处理中文/日文
            bool matched = false;
            // 尝试从最长的词组开始匹配
            for (size_t len = std::min(max_phrase_length, chars.size() - i); len > 0 && !matched; --len) {
                std::string phrase;
                for (size_t j = 0; j < len; ++j) {
                    phrase += chars[i + j];
                }
                
                if (lexicon.find(phrase) != lexicon.end()) {
                    auto& [phrase_phones, phrase_tones] = lexicon[phrase];
                    phones.insert(phones.end(), phrase_phones.begin(), phrase_phones.end());
                    tones.insert(tones.end(), phrase_tones.begin(), phrase_tones.end());
                    
                    // 打印匹配信息
                    std::cout << phrase << "\t|\t" << phonesToString(phrase_phones) << "\t|\t" 
                             << tonesToString(phrase_tones) << std::endl;
                    
                    i += len;
                    matched = true;
                    break;
                }
            }
            
            // 如果没有匹配到任何词组，使用'_'的发音
            if (!matched) {
                std::string c = chars[i++];
                std::string s = c;
                
                // 中文标点符号转换
                std::string orig_char = s; // 保留原始字符用于日志
                if (s == "，") s = ",";
                else if (s == "。") s = ".";
                else if (s == "！") s = "!";
                else if (s == "？") s = "?";

                // 如果词典中找不到，则使用'_'的发音
                if (lexicon.find(s) != lexicon.end()) {
                    auto& [char_phones, char_tones] = lexicon[s];
                    phones.insert(phones.end(), char_phones.begin(), char_phones.end());
                    tones.insert(tones.end(), char_tones.begin(), char_tones.end());
                    
                    // 打印匹配信息
                    std::cout << orig_char << "\t|\t" << phonesToString(char_phones) << "\t|\t" 
                             << tonesToString(char_tones) << std::endl;
                } else {
                    phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
                    tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());
                    
                    // 打印未匹配信息
                    std::cout << orig_char << "\t|\t" << phonesToString(unknown_token.first) << " (未匹配)\t|\t" 
                             << tonesToString(unknown_token.second) << std::endl;
                }
            }
        }
        
        // 在末尾添加'_'边界标记
        phones.insert(phones.end(), unknown_token.first.begin(), unknown_token.first.end());
        tones.insert(tones.end(), unknown_token.second.begin(), unknown_token.second.end());
        std::cout << "<EOS>\t|\t" << phonesToString(unknown_token.first) << "\t|\t" 
                 << tonesToString(unknown_token.second) << std::endl;
        
        // 汇总打印最终结果
        std::cout << "\n处理结果汇总:" << std::endl;
        std::cout << "原文: " << text << std::endl;
        std::cout << "音素: " << phonesToString(phones) << std::endl;
        std::cout << "声调: " << tonesToString(tones) << std::endl;
        std::cout << "====================" << std::endl;
    }

private:
    // 处理单个字符
    void processChar(const std::string& c, std::vector<int>& phones, std::vector<int>& tones) {
        std::string s = c;
        
        // 中文标点符号转换
        if (s == "，") s = ",";
        else if (s == "。") s = ".";
        else if (s == "！") s = "!";
        else if (s == "？") s = "?";

        // 如果词典中找不到，则使用'_'的发音
        auto& phones_and_tones = (lexicon.find(s) != lexicon.end()) ? lexicon[s] : unknown_token;
        
        phones.insert(phones.end(), phones_and_tones.first.begin(), phones_and_tones.first.end());
        tones.insert(tones.end(), phones_and_tones.second.begin(), phones_and_tones.second.end());
    }
    
    // 将音素ID数组转换为字符串用于日志输出
    std::string phonesToString(const std::vector<int>& phones) {
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
    
    // 将声调数组转换为字符串用于日志输出
    std::string tonesToString(const std::vector<int>& tones) {
        std::string result;
        for (auto tone : tones) {
            if (!result.empty()) result += " ";
            result += std::to_string(tone);
        }
        return result;
    }
};
