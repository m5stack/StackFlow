#pragma once

#include <fstream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept>

template <typename T>
void savetxt(const std::string& filename, const std::vector<T>& data, char delimiter = ' ', int precision = 6)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    std::ios_base::fmtflags original_flags = outfile.flags();
    if constexpr (std::is_floating_point_v<T>) {
        outfile << std::scientific << std::setprecision(precision);
    }
    if (!data.empty()) {
        outfile << data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            outfile << delimiter << data[i];
        }
    }
    outfile.flags(original_flags);
    outfile.close();
}

template <typename T>
int readtxt(const std::string& filename, std::vector<T>& data)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        T value;
        while (iss >> value) {
            data.push_back(value);
            if (iss.peek() == ',' || iss.peek() == ';') iss.ignore();
        }
    }
    file.close();
    return 0;
}

std::vector<float> linspace(float start, float end, std::size_t num_steps)
{
    if (num_steps == 0) {
        return {};
    }
    if (num_steps == 1) {
        return {start};
    }
    std::vector<float> result(num_steps);
    float step_size = (end - start) / (num_steps - 1);
    for (std::size_t i = 0; i < num_steps; ++i) {
        result[i] = start + i * step_size;
    }
    result.back() = end;
    return result;
}