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

const char* CLASS_NAMES[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const char* OBB_CLASS_NAMES[] = {"plane",
                                 "ship",
                                 "storage tank",
                                 "baseball diamond",
                                 "tennis court",
                                 "basketball court",
                                 "ground track field",
                                 "harbor",
                                 "bridge",
                                 "large vehicle",
                                 "small vehicle",
                                 "helicopter",
                                 "roundabout",
                                 "soccer ball field",
                                 "swimming pool"};

static const std::vector<std::vector<uint8_t>> COCO_COLORS = {
    {56, 0, 255},  {226, 255, 0}, {0, 94, 255},  {0, 37, 255},  {0, 255, 94},  {255, 226, 0}, {0, 18, 255},
    {255, 151, 0}, {170, 0, 255}, {0, 255, 56},  {255, 0, 75},  {0, 75, 255},  {0, 255, 169}, {255, 0, 207},
    {75, 255, 0},  {207, 0, 255}, {37, 0, 255},  {0, 207, 255}, {94, 0, 255},  {0, 255, 113}, {255, 18, 0},
    {255, 0, 56},  {18, 0, 255},  {0, 255, 226}, {170, 255, 0}, {255, 0, 245}, {151, 255, 0}, {132, 255, 0},
    {75, 0, 255},  {151, 0, 255}, {0, 151, 255}, {132, 0, 255}, {0, 255, 245}, {255, 132, 0}, {226, 0, 255},
    {255, 37, 0},  {207, 255, 0}, {0, 255, 207}, {94, 255, 0},  {0, 226, 255}, {56, 255, 0},  {255, 94, 0},
    {255, 113, 0}, {0, 132, 255}, {255, 0, 132}, {255, 170, 0}, {255, 0, 188}, {113, 255, 0}, {245, 0, 255},
    {113, 0, 255}, {255, 188, 0}, {0, 113, 255}, {255, 0, 0},   {0, 56, 255},  {255, 0, 113}, {0, 255, 188},
    {255, 0, 94},  {255, 0, 18},  {18, 255, 0},  {0, 255, 132}, {0, 188, 255}, {0, 245, 255}, {0, 169, 255},
    {37, 255, 0},  {255, 0, 151}, {188, 0, 255}, {0, 255, 37},  {0, 255, 0},   {255, 0, 170}, {255, 0, 37},
    {255, 75, 0},  {0, 0, 255},   {255, 207, 0}, {255, 0, 226}, {255, 245, 0}, {188, 255, 0}, {0, 255, 18},
    {0, 255, 75},  {0, 255, 151}, {255, 56, 0},  {245, 255, 0}};

static const std::vector<std::vector<uint8_t>> KPS_COLORS = {
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},   {255, 128, 0},
    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51, 153, 255},
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

static const std::vector<std::vector<uint8_t>> LIMB_COLORS = {
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};

static const std::vector<std::vector<uint8_t>> SKELETON = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
    {8, 10},  {9, 11},  {2, 3},   {1, 2},   {1, 3},   {2, 4},  {3, 5},  {4, 6}, {5, 7}};

void postprocess_yolo(const std::vector<std::vector<float>>& outputs, const cv::Mat& mat, int input_w, int input_h,
                      int cls_num, int point_num, float prob_threshold, float nms_threshold,
                      std::vector<detection::Object>& objects, const std::string& model_type)
{
    std::vector<detection::Object> proposals;

    if (model_type == "detect") {
        for (int i = 0; i < 3; ++i) {
            detection::generate_proposals_yolov8_native((1 << i) * 8, outputs[i].data(), prob_threshold, proposals,
                                                        input_w, input_h, cls_num);
        }
        detection::get_out_bbox(proposals, objects, nms_threshold, input_h, input_w, mat.rows, mat.cols);
    } else if (model_type == "segment") {
        for (int i = 0; i < 3; ++i) {
            int32_t stride       = (1 << i) * 8;
            const float* det_ptr = outputs[i].data();
            const float* seg_ptr = outputs[i + 3].data();
            generate_proposals_yolov8_seg_native(stride, det_ptr, seg_ptr, prob_threshold, proposals, input_w, input_h,
                                                 cls_num);
        }
        const float* mask_proto_ptr = outputs[6].data();
        get_out_bbox_mask(proposals, objects, mask_proto_ptr,
                          /*mask_proto_h*/ 32, /*mask_proto_w*/ 4, nms_threshold, input_h, input_w, mat.rows, mat.cols);
    } else if (model_type == "pose") {
        for (int i = 0; i < 3; ++i) {
            int32_t stride       = (1 << i) * 8;
            const float* det_ptr = outputs[i].data();
            const float* kps_ptr = outputs[i + 3].data();
            generate_proposals_yolov8_pose_native(stride, det_ptr, kps_ptr, prob_threshold, proposals, input_h, input_w,
                                                  point_num, cls_num);
        }
        get_out_bbox_kps(proposals, objects, nms_threshold, input_h, input_w, mat.rows, mat.cols);
    } else if (model_type == "obb") {
        std::vector<int> strides = {8, 16, 32};
        std::vector<detection::GridAndStride> grid_strides;
        detection::generate_grids_and_stride(input_w, input_h, strides, grid_strides);
        detection::obb::generate_proposals_yolov8_obb_native(grid_strides, outputs[0].data(), prob_threshold, proposals,
                                                             input_w, input_h, cls_num);
        detection::obb::get_out_obb_bbox(proposals, objects, nms_threshold, input_h, input_w, mat.rows, mat.cols);
    } else {
        SLOGE("Unsupported model type: %s", model_type.c_str());
    }
}

int AxclWrapper::post_process(cv::Mat& mat, int& input_w, int& input_h, int& cls_num, int& point_num,
                              float& prob_threshold, float& nms_threshold, std::vector<detection::Object>& objects,
                              std::string& model_type)
{
    if (!runner_) {
        SLOGE("Runner not initialized before post_process.");
        return -1;
    }

    int outputs_needed = 0;

    if (model_type == "detect")
        outputs_needed = 3;
    else if (model_type == "segment")
        outputs_needed = 7;
    else if (model_type == "pose")
        outputs_needed = 6;
    else if (model_type == "obb")
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

    postprocess_yolo(output_data, mat, input_w, input_h, cls_num, point_num, prob_threshold, nms_threshold, objects,
                     model_type);
    return 0;
}