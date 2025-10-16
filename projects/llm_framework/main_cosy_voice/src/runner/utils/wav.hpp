#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring> // for memcpy
#include <algorithm> // for std::clamp (C++17)

// 将 32 位整数以小端序写入文件
void write_little_endian_32(std::ofstream& file, uint32_t value) {
    file.put(static_cast<char>(value & 0xFF));
    file.put(static_cast<char>((value >> 8) & 0xFF));
    file.put(static_cast<char>((value >> 16) & 0xFF));
    file.put(static_cast<char>((value >> 24) & 0xFF));
}

// 将 16 位整数以小端序写入文件
void write_little_endian_16(std::ofstream& file, uint16_t value) {
    file.put(static_cast<char>(value & 0xFF));
    file.put(static_cast<char>((value >> 8) & 0xFF));
}

// 将 32 位浮点数以小端序写入文件 (IEEE 754)
void write_little_endian_float32(std::ofstream& file, float value) {
    // IEEE 754 float 在大多数系统上已经是小端序
    // 但为了确保跨平台兼容性，我们显式处理字节序
    static_assert(sizeof(float) == 4, "Float must be 32 bits");
    char bytes[4];
    std::memcpy(bytes, &value, 4);
    // bytes[0] 是最低有效字节 (LSB)
    file.put(bytes[0]);
    file.put(bytes[1]);
    file.put(bytes[2]);
    file.put(bytes[3]);
}

// 将 float 范围 [-1.0, 1.0] 转换为 16-bit signed integer [-32768, 32767]
int16_t float_to_int16(float sample) {
    // 确保输入在有效范围内
    // C++17 std::clamp, 或手动实现
    // sample = std::clamp(sample, -1.0f, 1.0f);
    if (sample > 1.0f) sample = 1.0f;
    if (sample < -1.0f) sample = -1.0f;
    
    // 转换
    return static_cast<int16_t>(sample * 32767.0f);
}


bool saveVectorAsWavFloat(const std::vector<float>& audio_data, const std::string& filename, int sample_rate, int channels) {
    if (audio_data.empty() || channels <= 0 || sample_rate <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    // --- 计算关键参数 ---
    uint32_t num_samples = static_cast<uint32_t>(audio_data.size());
    uint16_t bits_per_sample = 32; // IEEE Float
    uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
    uint16_t block_align = channels * (bits_per_sample / 8);
    uint32_t data_chunk_size = num_samples * (bits_per_sample / 8);
    uint32_t riff_chunk_size = 4 + (8 + 16) + (8 + data_chunk_size); // fmt chunk is 24 bytes including header

    // --- 写入 RIFF Header ---
    file.write("RIFF", 4);
    write_little_endian_32(file, riff_chunk_size);
    file.write("WAVE", 4);

    // --- 写入 fmt Subchunk ---
    file.write("fmt ", 4);
    write_little_endian_32(file, 16); // Subchunk1Size for PCM
    write_little_endian_16(file, 3); // AudioFormat: 3 = IEEE Float
    write_little_endian_16(file, static_cast<uint16_t>(channels));
    write_little_endian_32(file, sample_rate);
    write_little_endian_32(file, byte_rate);
    write_little_endian_16(file, block_align);
    write_little_endian_16(file, bits_per_sample);

    // --- 写入 data Subchunk ---
    file.write("data", 4);
    write_little_endian_32(file, data_chunk_size);

    // --- 写入音频数据 ---
    for (const float& sample : audio_data) {
        write_little_endian_float32(file, sample);
    }

    if (file.fail()) {
        std::cerr << "Error occurred while writing audio data." << std::endl;
        file.close();
        return false;
    }

    file.close();
    std::cout << "Successfully saved audio to " << filename << " (32-bit Float PCM)." << std::endl;
    return true;
}

bool saveVectorAsWavInt16(const std::vector<float>& audio_data, const std::string& filename, int sample_rate, int channels) {
    if (audio_data.empty() || channels <= 0 || sample_rate <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    // --- 计算关键参数 ---
    uint32_t num_samples = static_cast<uint32_t>(audio_data.size());
    uint16_t bits_per_sample = 16; // 16-bit signed integer
    uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
    uint16_t block_align = channels * (bits_per_sample / 8);
    uint32_t data_chunk_size = num_samples * (bits_per_sample / 8);
    uint32_t riff_chunk_size = 4 + (8 + 16) + (8 + data_chunk_size); // fmt chunk is 24 bytes including header

    // --- 写入 RIFF Header ---
    file.write("RIFF", 4);
    write_little_endian_32(file, riff_chunk_size);
    file.write("WAVE", 4);

    // --- 写入 fmt Subchunk ---
    file.write("fmt ", 4);
    write_little_endian_32(file, 16); // Subchunk1Size for PCM
    write_little_endian_16(file, 1); // AudioFormat: 1 = PCM (Integer)
    write_little_endian_16(file, static_cast<uint16_t>(channels));
    write_little_endian_32(file, sample_rate);
    write_little_endian_32(file, byte_rate);
    write_little_endian_16(file, block_align);
    write_little_endian_16(file, bits_per_sample);

    // --- 写入 data Subchunk ---
    file.write("data", 4);
    write_little_endian_32(file, data_chunk_size);

    // --- 写入音频数据 ---
    for (const float& sample : audio_data) {
        int16_t int_sample = float_to_int16(sample);
        write_little_endian_16(file, static_cast<uint16_t>(int_sample)); // int16_t -> uint16_t for raw bytes
    }

    if (file.fail()) {
        std::cerr << "Error occurred while writing audio data." << std::endl;
        file.close();
        return false;
    }

    file.close();
    std::cout << "Successfully saved audio to " << filename << " (16-bit Integer PCM)." << std::endl;
    return true;
}


