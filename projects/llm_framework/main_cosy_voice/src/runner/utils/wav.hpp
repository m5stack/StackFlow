#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

void write_little_endian_32(std::ofstream& file, uint32_t value)
{
    file.put(static_cast<char>(value & 0xFF));
    file.put(static_cast<char>((value >> 8) & 0xFF));
    file.put(static_cast<char>((value >> 16) & 0xFF));
    file.put(static_cast<char>((value >> 24) & 0xFF));
}

void write_little_endian_16(std::ofstream& file, uint16_t value)
{
    file.put(static_cast<char>(value & 0xFF));
    file.put(static_cast<char>((value >> 8) & 0xFF));
}

void write_little_endian_float32(std::ofstream& file, float value)
{
    static_assert(sizeof(float) == 4, "Float must be 32 bits");
    char bytes[4];
    std::memcpy(bytes, &value, 4);
    file.put(bytes[0]);
    file.put(bytes[1]);
    file.put(bytes[2]);
    file.put(bytes[3]);
}

int16_t float_to_int16(float sample)
{
    if (sample > 1.0f) sample = 1.0f;
    if (sample < -1.0f) sample = -1.0f;
    return static_cast<int16_t>(sample * 32767.0f);
}

bool ensure_directory_exists(const std::string& filename)
{
    try {
        fs::path filepath(filename);
        fs::path dir = filepath.parent_path();
        if (!dir.empty() && !fs::exists(dir)) {
            fs::create_directories(dir);
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Failed to create directories: " << e.what() << std::endl;
        return false;
    }
}

bool saveVectorAsWavFloat(const std::vector<float>& audio_data, const std::string& filename, int sample_rate,
                          int channels)
{
    if (audio_data.empty() || channels <= 0 || sample_rate <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return false;
    }

    if (!ensure_directory_exists(filename)) return false;

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    uint32_t num_samples     = static_cast<uint32_t>(audio_data.size());
    uint16_t bits_per_sample = 32;  // IEEE Float
    uint32_t byte_rate       = sample_rate * channels * (bits_per_sample / 8);
    uint16_t block_align     = channels * (bits_per_sample / 8);
    uint32_t data_chunk_size = num_samples * (bits_per_sample / 8);
    uint32_t riff_chunk_size = 4 + (8 + 16) + (8 + data_chunk_size);

    file.write("RIFF", 4);
    write_little_endian_32(file, riff_chunk_size);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    write_little_endian_32(file, 16);
    write_little_endian_16(file, 3);
    write_little_endian_16(file, static_cast<uint16_t>(channels));
    write_little_endian_32(file, sample_rate);
    write_little_endian_32(file, byte_rate);
    write_little_endian_16(file, block_align);
    write_little_endian_16(file, bits_per_sample);
    file.write("data", 4);
    write_little_endian_32(file, data_chunk_size);

    for (const float& sample : audio_data) {
        write_little_endian_float32(file, sample);
    }

    if (file.fail()) {
        std::cerr << "Error occurred while writing audio data." << std::endl;
        return false;
    }

    file.close();
    std::cout << "Successfully saved audio to " << filename << " (32-bit Float PCM)." << std::endl;
    return true;
}

bool saveVectorAsWavInt16(const std::vector<float>& audio_data, const std::string& filename, int sample_rate,
                          int channels)
{
    if (audio_data.empty() || channels <= 0 || sample_rate <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return false;
    }

    if (!ensure_directory_exists(filename)) return false;

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    uint32_t num_samples     = static_cast<uint32_t>(audio_data.size());
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate       = sample_rate * channels * (bits_per_sample / 8);
    uint16_t block_align     = channels * (bits_per_sample / 8);
    uint32_t data_chunk_size = num_samples * (bits_per_sample / 8);
    uint32_t riff_chunk_size = 4 + (8 + 16) + (8 + data_chunk_size);

    file.write("RIFF", 4);
    write_little_endian_32(file, riff_chunk_size);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    write_little_endian_32(file, 16);
    write_little_endian_16(file, 1);
    write_little_endian_16(file, static_cast<uint16_t>(channels));
    write_little_endian_32(file, sample_rate);
    write_little_endian_32(file, byte_rate);
    write_little_endian_16(file, block_align);
    write_little_endian_16(file, bits_per_sample);
    file.write("data", 4);
    write_little_endian_32(file, data_chunk_size);

    for (const float& sample : audio_data) {
        int16_t int_sample = float_to_int16(sample);
        write_little_endian_16(file, static_cast<uint16_t>(int_sample));
    }

    if (file.fail()) {
        std::cerr << "Error occurred while writing audio data." << std::endl;
        return false;
    }

    file.close();
    std::cout << "Successfully saved audio to " << filename << " (16-bit Integer PCM)." << std::endl;
    return true;
}