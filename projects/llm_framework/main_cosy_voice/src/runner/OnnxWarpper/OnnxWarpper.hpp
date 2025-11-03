#pragma once
#include "../BaseRunner.hpp"

#include "onnxruntime_cxx_api.h"

class OnnxRunner : virtual public BaseRunner
{
    Ort::Env env;
    Ort::Session session{nullptr};

    std::vector<std::vector<size_t>> inputs_shape;
    std::vector<std::string> inputs_name;
    std::vector<const char *> inputs_name_cstr;
    std::vector<std::shared_ptr<float>> inputs_data;
    std::vector<Ort::Value> inputs_tensor;

    std::vector<std::vector<size_t>> outputs_shape;
    std::vector<std::string> outputs_name;
    std::vector<const char *> outputs_name_cstr;
    std::vector<std::shared_ptr<float>> outputs_data;
    std::vector<Ort::Value> outputs_tensor;

public:
    int load(BaseConfig &config) override
    {
        Ort::SessionOptions session_options;
        session_options.SetInterOpNumThreads(config.nthread);
        session_options.SetIntraOpNumThreads(config.nthread);
        // session_options
        // TensorRT加速开启，CUDA加速开启
        // OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0); // tensorRT
        // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::AllocatorWithDefaultOptions allocator;
        session = Ort::Session{env, config.onnx_model.c_str(), session_options};
        printf("\ninputs: \n");
        for (size_t i = 0; i < session.GetInputCount(); i++)
        {
            auto input_name = std::string(session.GetInputNameAllocated(i, allocator).get());
            inputs_name.push_back(input_name);

            printf("%20s: ", input_name.c_str());
            auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

            if (input_shape.size() && input_shape[0] < 0)
            {
                input_shape[0] = 1;
            }
            std::vector<size_t> tmp_input_shape(input_shape.size());
            for (size_t j = 0; j < input_shape.size(); j++)
            {
                tmp_input_shape[j] = input_shape[j];
                printf("%ld", tmp_input_shape[j]);
                if (j < (input_shape.size() - 1))
                    printf(" x ");
            }
            printf("\n");
            inputs_shape.push_back(tmp_input_shape);

            int len = 1;
            for (size_t d = 0; d < input_shape.size(); d++)
            {
                len *= input_shape[d];
            }
            std::shared_ptr<float> data(new float[len], std::default_delete<float[]>());
            inputs_data.push_back(data);

            inputs_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, data.get(), len, input_shape.data(), input_shape.size()));
        }
        printf("output: \n");
        for (size_t i = 0; i < session.GetOutputCount(); i++)
        {
            auto output_name = std::string(session.GetOutputNameAllocated(i, allocator).get());
            outputs_name.push_back(output_name);

            printf("%20s: ", output_name.c_str());
            auto output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (output_shape.size() && output_shape[0] < 0)
            {
                output_shape[0] = 1;
            }
            std::vector<size_t> tmp_output_shape(output_shape.size());
            for (size_t j = 0; j < output_shape.size(); j++)
            {
                tmp_output_shape[j] = output_shape[j];
                printf("%ld", tmp_output_shape[j]);
                if (j < (output_shape.size() - 1))
                    printf(" x ");
            }
            printf("\n");
            outputs_shape.push_back(tmp_output_shape);

            int len = 1;
            for (size_t d = 0; d < output_shape.size(); d++)
            {
                len *= output_shape[d];
            }
            std::shared_ptr<float> data(new float[len], std::default_delete<float[]>());
            outputs_data.push_back(data);

            outputs_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, data.get(), len, output_shape.data(), output_shape.size()));
        }

        inputs_name_cstr.resize(inputs_name.size());
        for (size_t i = 0; i < inputs_name.size(); i++)
        {
            inputs_name_cstr[i] = inputs_name[i].c_str();
        }

        outputs_name_cstr.resize(outputs_name.size());
        for (size_t i = 0; i < outputs_name.size(); i++)
        {
            outputs_name_cstr[i] = outputs_name[i].c_str();
        }

        return 0;
    }

    int inference() override
    {
        Ort::RunOptions run_options;
        // for (size_t i = 0; i < inputs_name_cstr.size(); i++)
        // {
        //     printf("%20s: \n", inputs_name_cstr[i]);
        // }
        //  for (size_t i = 0; i < outputs_name_cstr.size(); i++)
        // {
        //     printf("%20s: \n", outputs_name_cstr[i]);
        // }
        session.Run(run_options, inputs_name_cstr.data(), inputs_tensor.data(), inputs_tensor.size(), outputs_name_cstr.data(), outputs_tensor.data(), outputs_tensor.size());
        return 0;
    }

    int getInputCount() override
    {
        return inputs_shape.size();
    }

    std::vector<size_t> getInputShape(int idx) override
    {
        return inputs_shape[idx];
    }

    std::string getInputName(int idx) override
    {
        return inputs_name[idx];
    }

    float *getInputPtr(int idx) override
    {
        return inputs_data[idx].get();
    }

    int getOutputCount() override
    {
        return outputs_shape.size();
    }

    std::vector<size_t> getOutputShape(int idx) override
    {
        return outputs_shape[idx];
    }

    std::string getOutputName(int idx) override
    {
        return outputs_name[idx];
    }

    float *getOutputPtr(int idx) override
    {
        return outputs_data[idx].get();
    }
};
