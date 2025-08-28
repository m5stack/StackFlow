#include "base/detection.hpp"
#define UNUSE_STRUCT_OBJECT
#include "AxclWrapper.h"

#include "../../../../SDK/components/utilities/include/sample_log.h"

AxclWrapper::AxclWrapper() : runner_(std::make_unique<middleware::runtime_runner>())
{
}

AxclWrapper::~AxclWrapper()
{
    finalize();
}

bool AxclWrapper::initialize(const std::string& configFile, uint32_t index, uint32_t kind, const std::string& modelPath,
                             bool input_cached, bool output_cached, uint32_t group, uint32_t batch)
{
    // runner_ = std::make_unique<middleware::runtime_runner>();
    if (!runner_) {
        SLOGE("Failed to create runner instance.");
        return false;
    }

    if (!runner_->init(configFile, index, kind)) {
        SLOGE("Init failed for runner with config {%s}, index {%u}, kind {%u}.", configFile.c_str(), index, kind);
        runner_.reset();
        return false;
    }

    if (!runner_->load(modelPath)) {
        SLOGE("Loading model {%s} failed.", modelPath.c_str());
        return false;
    }

    if (!runner_->prepare(input_cached, output_cached, group, batch)) {
        SLOGE("Prepare failed for runner with input_cached {%s}, output_cached {%s}, group {%u}, batch {%u}.",
              input_cached ? "true" : "false", output_cached ? "true" : "false", group, batch);
        runner_.reset();
        return false;
    }

    return true;
}

bool AxclWrapper::run(bool sync)
{
    if (!runner_) {
        SLOGE("Runner not initialized before running.");
        return false;
    }
    return runner_->run(sync);
}

bool AxclWrapper::set()
{
    if (!runner_) {
        SLOGE("Runner not initialized before running.");
        return false;
    }
    return runner_->set();
}

size_t AxclWrapper::getInputSize(int index) const
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting input size for index %d.", index);
        return 0;
    }
    return runner_->get_input_size(index);
}

size_t AxclWrapper::getOutputSize(int index) const
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting output size for index %d.", index);
        return 0;
    }
    return runner_->get_output_size(index);
}

void* AxclWrapper::getInputPointer(int index)
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting input pointer for index %d.", index);
        return nullptr;
    }
    return runner_->get_input_pointer(index);
}

void* AxclWrapper::getOutputPointer(int index)
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting output pointer for index %d.", index);
        return nullptr;
    }
    return runner_->get_output_pointer(index);
}

void AxclWrapper::finalize()
{
    if (runner_) {
        runner_->final();
        runner_.reset();
        SLOGI("AxclWrapper for model {%s} finalized.", model_path_.c_str());
    }
}

void postprocess_seg(const std::vector<std::vector<float>>& outputs, int model_input_h, int model_input_w, const cv::Mat& mat, const std::string& model_type, std::string &byteString)
{
    const std::vector<float>& mask_proto_vec = outputs[0];
    cv::Mat feature(model_input_h, model_input_w, CV_32FC1, const_cast<float*>(mask_proto_vec.data()));

    double minVal, maxVal;
    cv::minMaxLoc(feature, &minVal, &maxVal);

    feature -= minVal;
    feature /= (maxVal - minVal);
    feature *= 255.0;
    feature.convertTo(feature, CV_8UC1);

    cv::Mat color_map(feature.rows, feature.cols, CV_8UC3);
    cv::applyColorMap(feature, color_map, cv::COLORMAP_MAGMA);
    cv::resize(color_map, color_map, cv::Size(mat.cols, mat.rows));

    std::vector<uchar> buf;
    cv::imencode(".jpg", color_map, buf);
    byteString.assign(buf.begin(), buf.end());
}

int AxclWrapper::post_process(cv::Mat &mat, int model_input_h, int model_input_w, std::string &model_type, std::string &byteString)
{
    if (!runner_) {
        SLOGE("Runner not initialized before post_process.");
        return -1;
    }

    int outputs_needed = 0;

    if (model_type == "segment")
        outputs_needed = 1;
    else {
        SLOGE("Unknown model type: %s", model_type.c_str());
        return -1;
    }

    std::vector<std::vector<float>> output_data(outputs_needed);
    for (int i = 0; i < outputs_needed; ++i) {
        size_t sz_bytes = getOutputSize(i);
        output_data[i].resize(sz_bytes / sizeof(float));
        axclrtMemcpy(output_data[i].data(), getOutputPointer(i), sz_bytes, AXCL_MEMCPY_DEVICE_TO_HOST);
    }

    postprocess_seg(output_data, model_input_h, model_input_w, mat, model_type, byteString);
    return 0;
}