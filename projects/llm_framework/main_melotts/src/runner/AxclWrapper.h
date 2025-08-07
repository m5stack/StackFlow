#pragma once

#include <memory>
#include <string>
#include <vector>
#include <axcl.h>
#include "middleware/axcl_runtime_runner.hpp"

class AxclWrapper {
public:
    AxclWrapper();
    ~AxclWrapper();

    AxclWrapper(const AxclWrapper&)            = delete;
    AxclWrapper& operator=(const AxclWrapper&) = delete;

    bool initialize(const std::string& configFile, uint32_t index, uint32_t kind, const std::string& modelPath,
                    bool input_cached, bool output_cached, uint32_t group, uint32_t batch);

    bool run(bool sync);
    bool set();

    size_t getInputSize(int index) const;
    size_t getOutputSize(int index) const;

    void* getInputPointer(int index);
    void* getOutputPointer(int index);

    void finalize();

private:
    std::unique_ptr<middleware::runner> runner_;
    std::string model_path_;
};