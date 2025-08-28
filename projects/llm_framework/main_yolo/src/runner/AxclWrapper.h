#pragma once

#include <memory>
#include <string>
#include <vector>
#include <axcl.h>

#include <opencv2/opencv.hpp>
#include "middleware/axcl_runtime_runner.hpp"

#ifndef UNUSE_STRUCT_OBJECT
namespace detection {
typedef struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Point2f landmark[5];
    /* for yolov5-seg */
    cv::Mat mask;
    std::vector<float> mask_feat;
    std::vector<float> kps_feat;
    /* for yolov8-obb */
    float angle;
} Object;

}  // namespace detection
#endif

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

    int post_process(cv::Mat& mat, int& input_w, int& input_, int& cls_num, int& point_num, float& pron_threshold,
                     float& nms_threshold, std::vector<detection::Object>& objects, std::string& model_type);


private:
    std::unique_ptr<middleware::runner> runner_;
    std::string model_path_;
};