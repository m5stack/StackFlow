#pragma once
#include "string"
#include "vector"
#include "memory"

enum RunnerType
{
    RT_UNKNOWN,
    RT_OnnxRunner,
    RT_END,
};

struct BaseConfig
{
    std::string onnx_model;
    std::string output_model; // for trt
    int nthread;
};

class BaseRunner
{
public:
    virtual int load(BaseConfig &config) = 0;
    virtual int inference() = 0;

    virtual int getInputCount() = 0;
    virtual std::vector<size_t> getInputShape(int idx) = 0;
    virtual std::string getInputName(int idx) = 0;
    virtual float *getInputPtr(int idx) = 0;

    virtual int getOutputCount() = 0;
    virtual std::vector<size_t> getOutputShape(int idx) = 0;
    virtual std::string getOutputName(int idx) = 0;
    virtual float *getOutputPtr(int idx) = 0;
};

std::shared_ptr<BaseRunner> CreateRunner(RunnerType rt);
