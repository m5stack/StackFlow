/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/
#include "base/detection.hpp"
#define UNUSE_STRUCT_OBJECT
#include "EngineWrapper.hpp"
#include "utils/io.hpp"
#include <cstdlib>

#include <global_config.h>

#if defined(CONFIG_AX_650C_MSP_ENABLED)
static const char* strAlgoModelType[AX_ENGINE_MODEL_TYPE_BUTT] = {"3.6T", "7.2T", "18T"};
#endif

#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
static const char* strAlgoModelType[AX_ENGINE_MODEL_TYPE_BUTT] = {"HalfOCM", "FullOCM"};
#endif

/// @brief npu type
typedef enum axNPU_TYPE_E {
    AX_NPU_DEFAULT = 0,        /* running under default NPU according to system */
    AX_STD_VNPU_1  = (1 << 0), /* running under STD VNPU1 */
    AX_STD_VNPU_2  = (1 << 1), /* running under STD VNPU2 */
    AX_STD_VNPU_3  = (1 << 2), /* running under STD VNPU3 */
    AX_BL_VNPU_1   = (1 << 3), /* running under BIG-LITTLE VNPU1 */
    AX_BL_VNPU_2   = (1 << 4)  /* running under BIG-LITTLE VNPU2 */
} AX_NPU_TYPE_E;

#if defined(CONFIG_AX_650C_MSP_ENABLED)
static AX_S32 CheckModelVNpu(const std::string& strModel, const AX_ENGINE_MODEL_TYPE_T& eModelType,
                             const AX_S32& nNpuType, AX_U32& nNpuSet)
{
    AX_ENGINE_NPU_ATTR_T stNpuAttr;
    memset(&stNpuAttr, 0x00, sizeof(stNpuAttr));

    auto ret = AX_ENGINE_GetVNPUAttr(&stNpuAttr);
    if (ret == 0) {
        // VNPU DISABLE
        if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_DISABLE) {
            nNpuSet = 0x01;  // NON-VNPU (0b111)
            // printf("%s will run under VNPU-DISABLE [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
        }
        // STD VNPU
        else if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_STD) {
            // 7.2T & 10.8T no allow
            if (eModelType == AX_ENGINE_MODEL_TYPE1 || eModelType == AX_ENGINE_MODEL_TYPE2) {
                // printf("%s model type%d: [%s], no allow run under STD VNPU\n", strModel.c_str(), eModelType,
                // strAlgoModelType[eModelType]);
                return -1;
            }

            // default STD VNPU2
            if (nNpuType == 0) {
                nNpuSet = 0x02;  // VNPU2 (0b010)
                // printf("%s will run under default STD-VNPU2 [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
            } else {
                if (nNpuType & AX_STD_VNPU_1) {
                    nNpuSet |= 0x01;  // VNPU1 (0b001)
                    // printf("%s will run under STD-VNPU1 [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
                }
                if (nNpuType & AX_STD_VNPU_2) {
                    nNpuSet |= 0x02;  // VNPU2 (0b010)
                    // printf("%s will run under STD-VNPU2 [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
                }
                if (nNpuType & AX_STD_VNPU_3) {
                    nNpuSet |= 0x04;  // VNPU3 (0b100)
                    // printf("%s will run under STD-VNPU3 [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
                }
            }
        }
        // BL VNPU
        else if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE) {
            // 10.8T no allow
            if (eModelType == AX_ENGINE_MODEL_TYPE2) {
                // printf("%s model type%d: [%s], no allow run under BL VNPU\n", strModel.c_str(), eModelType,
                // strAlgoModelType[eModelType]);
                return -1;
            }

            // default BL VNPU
            if (nNpuType == 0) {
                // 7.2T default BL VNPU1
                if (eModelType == AX_ENGINE_MODEL_TYPE1) {
                    nNpuSet = 0x01;  // VNPU1 (0b001)
                    // printf("%s will run under default BL-VNPU1 [%s]\n", strModel.c_str(),
                    // strAlgoModelType[eModelType]);
                }
                // 3.6T default BL VNPU2
                else {
                    nNpuSet = 0x02;  // VNPU2 (0b010)
                    // printf("%s will run under default BL-VNPU2 [%s]\n", strModel.c_str(),
                    // strAlgoModelType[eModelType]);
                }
            } else {
                // 7.2T
                if (eModelType == AX_ENGINE_MODEL_TYPE1) {
                    // no allow set to BL VNPU2
                    if (nNpuType & AX_BL_VNPU_2) {
                        // printf("%s model type%d: [%s], no allow run under BL VNPU2\n", strModel.c_str(), eModelType,
                        // strAlgoModelType[eModelType]);
                        return -1;
                    }
                    if (nNpuType & AX_BL_VNPU_1) {
                        nNpuSet |= 0x01;  // VNPU1 (0b001)
                        // printf("%s will run under BL-VNPU1 [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
                    }
                }
                // 3.6T
                else {
                    if (nNpuType & AX_BL_VNPU_1) {
                        nNpuSet |= 0x01;  // VNPU1 (0b001)
                        // printf("%s will run under BL-VNPU1 [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
                    }
                    if (nNpuType & AX_BL_VNPU_2) {
                        nNpuSet |= 0x02;  // VNPU2 (0b010)
                        // printf("%s will run under BL-VNPU2 [%s]\n", strModel.c_str(), strAlgoModelType[eModelType]);
                    }
                }
            }
        }
    } else {
        printf("AX_ENGINE_GetVNPUAttr fail ret = %x\n", ret);
    }

    return ret;
}
#endif

#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
static AX_S32 CheckModelVNpu(const std::string& strModel, const AX_ENGINE_MODEL_TYPE_T& eModelType,
                             const AX_S32& nNpuType, AX_U32& nNpuSet)
{
    AX_ENGINE_NPU_ATTR_T stNpuAttr;
    memset(&stNpuAttr, 0x00, sizeof(stNpuAttr));

    auto ret = AX_ENGINE_GetVNPUAttr(&stNpuAttr);
    if (ret == 0) {
        // VNPU DISABLE
        if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_DISABLE) {
            nNpuSet = 0x01;  // NON-VNPU (0b111)
            // ALOGN("%s will run under VNPU-DISABLE [%s]", strModel.c_str(), strAlgoModelType[eModelType]);
        }
        // STD VNPU
        else if (stNpuAttr.eHardMode == AX_ENGINE_VIRTUAL_NPU_ENABLE) {
            // full ocm model was no allowned
            if (eModelType == AX_ENGINE_MODEL_TYPE1) {
                // printf("%s model type%d: [%s], no allow run under STD VNPU", strModel.c_str(), eModelType,
                // strAlgoModelType[eModelType]);
                return -1;
            }

            // default STD VNPU2
            if (nNpuType == 0) {
                nNpuSet = 0x02;  // VNPU2 (0b010)
                // printf("%s will run under default STD-VNPU2 [%s]", strModel.c_str(), strAlgoModelType[eModelType]);
            } else {
                if (nNpuType & AX_STD_VNPU_1) {
                    nNpuSet |= 0x01;  // VNPU1 (0b001)
                    // printf("%s will run under STD-VNPU1 [%s]", strModel.c_str(), strAlgoModelType[eModelType]);
                }
                if (nNpuType & AX_STD_VNPU_2) {
                    nNpuSet |= 0x02;  // VNPU2 (0b010)
                    // printf("%s will run under STD-VNPU2 [%s]", strModel.c_str(), strAlgoModelType[eModelType]);
                }
            }
        }
    } else {
        printf("AX_ENGINE_GetVNPUAttr fail ret = %x", ret);
    }

    return ret;
}
#endif

int EngineWrapper::Init(const char* strModelPath, uint32_t nNpuType, uint32_t npuMode)
{
    AX_S32 ret = 0;

    // 0. Init AX_ENGINE
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(npuMode);
    ret                = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
    }

    // 1. load model
    AX_BOOL bLoadModelUseCmm     = AX_TRUE;
    AX_CHAR* pModelBufferVirAddr = nullptr;
    AX_U64 u64ModelBufferPhyAddr = 0;
    AX_U32 nModelBufferSize      = 0;

    std::vector<char> model_buffer;

    if (bLoadModelUseCmm) {
        if (!utils::read_file(strModelPath, (AX_VOID**)&pModelBufferVirAddr, u64ModelBufferPhyAddr, nModelBufferSize)) {
            printf("ALGO read model(%s) fail\n", strModelPath);
            return -1;
        }
    } else {
        if (!utils::read_file(strModelPath, model_buffer)) {
            printf("ALGO read model(%s) fail\n", strModelPath);
            return -1;
        }

        pModelBufferVirAddr = model_buffer.data();
        nModelBufferSize    = model_buffer.size();
    }

    auto freeModelBuffer = [&]() {
        if (bLoadModelUseCmm) {
            if (u64ModelBufferPhyAddr != 0) {
                AX_SYS_MemFree(u64ModelBufferPhyAddr, &pModelBufferVirAddr);
            }
        } else {
            std::vector<char>().swap(model_buffer);
        }
        return;
    };

    // 1.1 Get Model Type
    AX_ENGINE_MODEL_TYPE_T eModelType = AX_ENGINE_MODEL_TYPE0;
    ret                               = AX_ENGINE_GetModelType(pModelBufferVirAddr, nModelBufferSize, &eModelType);
    if (0 != ret || eModelType >= AX_ENGINE_MODEL_TYPE_BUTT) {
        printf("%s AX_ENGINE_GetModelType fail ret=%x, eModelType=%d\n", strModelPath, eModelType);
        freeModelBuffer();
        return -1;
    }

    // 1.2 Check VNPU
    AX_ENGINE_NPU_SET_T nNpuSet = 0;
    ret                         = CheckModelVNpu(strModelPath, eModelType, nNpuType, nNpuSet);
    if (0 != ret) {
        printf("ALGO CheckModelVNpu fail\n");
        freeModelBuffer();
        return -1;
    }

    // 2. create handle
    AX_ENGINE_HANDLE handle = nullptr;
    ret                     = AX_ENGINE_CreateHandle(&handle, pModelBufferVirAddr, nModelBufferSize);
    auto deinit_handle      = [&handle]() {
        if (handle) {
            AX_ENGINE_DestroyHandle(handle);
        }
        return -1;
    };

    freeModelBuffer();

    if (0 != ret || !handle) {
        printf("ALGO Create model(%s) handle fail\n", strModelPath);

        return deinit_handle();
    }

    // 3. create context
    ret = AX_ENGINE_CreateContext(handle);
    if (0 != ret) {
        return deinit_handle();
    }

    // 4. set io
    m_io_info = nullptr;
    ret       = AX_ENGINE_GetIOInfo(handle, &m_io_info);
    if (0 != ret) {
        return deinit_handle();
    }
    m_input_num  = m_io_info->nInputSize;
    m_output_num = m_io_info->nOutputSize;

    // 4.1 query io
    //    AX_IMG_FORMAT_E eDtype;
    //    ret = utils::query_model_input_size(m_io_info, m_input_size, eDtype);//FIXME.
    //    if (0 != ret) {
    //        printf("model(%s) query model input size fail\n", strModelPath.c_str());
    //        return deinit_handle();
    //    }

    //    if (!(eDtype == AX_FORMAT_YUV420_SEMIPLANAR ||  eDtype == AX_FORMAT_YUV420_SEMIPLANAR_VU ||
    //        eDtype == AX_FORMAT_RGB888 || eDtype == AX_FORMAT_BGR888)) {
    //        printf("model(%s) data type is: 0x%02X, unsupport\n", strModelPath, eDtype);
    //        return deinit_handle();
    //    }

    // 4.2 brief io
#ifdef __DEBUG__
    printf("brief_io_info\n");
    utils::brief_io_info(strModelPath, m_io_info);
#endif

    // 5. Config VNPU
    //  printf("model(%s) nNpuSet: 0x%08X\n", strModelPath.c_str(), nNpuSet);
    //  will do nothing for using create handle v2 api

    // 6. prepare io
    // AX_U32 nIoDepth = (stCtx.vecOutputBufferFlag.size() == 0) ? 1 : stCtx.vecOutputBufferFlag.size();
    ret = utils::prepare_io(strModelPath, m_io_info, m_io, utils::IO_BUFFER_STRATEGY_CACHED);
    if (0 != ret) {
        printf("prepare io failed!\n");
        utils::free_io(m_io);
        return deinit_handle();
    }

    m_handle  = handle;
    m_hasInit = true;

    return 0;
}

int EngineWrapper::SetInput(void* pInput, int index)
{
    return utils::push_io_input(pInput, index, m_io);
}

int EngineWrapper::Run()
{
    if (!m_hasInit) return -1;

    // 7.3 run & benchmark
    auto ret = AX_ENGINE_RunSync(m_handle, &m_io);
    if (0 != ret) {
        printf("AX_ENGINE_RunSync failed. ret=0x%x\n", ret);
        return ret;
    }

    return 0;
}

int EngineWrapper::GetOutput(void* pOutput, int index)
{
    return utils::push_io_output(pOutput, index, m_io);
}

int EngineWrapper::GetInputSize(int index)
{
    return m_io.pInputs[index].nSize;
}

int EngineWrapper::GetOutputSize(int index)
{
    return m_io.pOutputs[index].nSize;
}

void* EngineWrapper::GetOutputPtr(int index)
{
    utils::cache_io_flush(&m_io.pOutputs[index]);
    return m_io.pOutputs[index].pVirAddr;
}

int EngineWrapper::Release()
{
    if (m_handle) {
        utils::free_io(m_io);
        AX_ENGINE_DestroyHandle(m_handle);
        m_handle = nullptr;
    }
    return 0;
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

void post_process(AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data, const cv::Mat& mat, int& input_w, int& input_h,
                  int& cls_num, int& point_num, float& prob_threshold, float& nms_threshold,
                  std::vector<detection::Object>& objects, std::string& model_type)
{
    // std::vector<detection::Object> objects;
    std::vector<detection::Object> proposals;
    if (model_type == "detect") {
        for (int i = 0; i < 3; ++i) {
            auto feat_ptr  = (float*)io_data->pOutputs[i].pVirAddr;
            int32_t stride = (1 << i) * 8;
            detection::generate_proposals_yolov8_native(stride, feat_ptr, prob_threshold, proposals, input_w, input_h,
                                                        cls_num);
        }
        detection::get_out_bbox(proposals, objects, nms_threshold, input_h, input_w, mat.rows, mat.cols);
        // detection::draw_objects(mat, objects, CLASS_NAMES, "yolo11_out");
    } else if (model_type == "segment") {
        float* output_ptr[3]     = {(float*)io_data->pOutputs[0].pVirAddr, (float*)io_data->pOutputs[1].pVirAddr,
                                    (float*)io_data->pOutputs[2].pVirAddr};
        float* output_seg_ptr[3] = {(float*)io_data->pOutputs[3].pVirAddr, (float*)io_data->pOutputs[4].pVirAddr,
                                    (float*)io_data->pOutputs[5].pVirAddr};
        for (int i = 0; i < 3; ++i) {
            auto feat_ptr     = output_ptr[i];
            auto feat_seg_ptr = output_seg_ptr[i];
            int32_t stride    = (1 << i) * 8;
            detection::generate_proposals_yolov8_seg_native(stride, feat_ptr, feat_seg_ptr, prob_threshold, proposals,
                                                            input_w, input_h, cls_num);
        }
        auto mask_proto_ptr = (float*)io_data->pOutputs[6].pVirAddr;
        detection::get_out_bbox_mask(proposals, objects, mask_proto_ptr, 32, 4, nms_threshold, input_h, input_w,
                                     mat.rows, mat.cols);
        // detection::draw_objects_mask(mat, objects, CLASS_NAMES, COCO_COLORS, "yolo11_seg_out");
    } else if (model_type == "pose") {
        float* output_ptr[3]     = {(float*)io_data->pOutputs[0].pVirAddr, (float*)io_data->pOutputs[1].pVirAddr,
                                    (float*)io_data->pOutputs[2].pVirAddr};
        float* output_kps_ptr[3] = {(float*)io_data->pOutputs[3].pVirAddr, (float*)io_data->pOutputs[4].pVirAddr,
                                    (float*)io_data->pOutputs[5].pVirAddr};

        for (int i = 0; i < 3; ++i) {
            auto feat_ptr     = output_ptr[i];
            auto feat_kps_ptr = output_kps_ptr[i];
            int32_t stride    = (1 << i) * 8;
            detection::generate_proposals_yolov8_pose_native(stride, feat_ptr, feat_kps_ptr, prob_threshold, proposals,
                                                             input_h, input_w, point_num, cls_num);
        }
        detection::get_out_bbox_kps(proposals, objects, nms_threshold, input_h, input_w, mat.rows, mat.cols);
        // detection::draw_keypoints(mat, objects, KPS_COLORS, LIMB_COLORS, SKELETON, "yolo11_pose_out");
    } else if (model_type == "obb") {
        std::vector<int> strides = {8, 16, 32};
        std::vector<detection::GridAndStride> grid_strides;
        detection::generate_grids_and_stride(input_w, input_h, strides, grid_strides);
        auto feat_ptr = (float*)io_data->pOutputs[0].pVirAddr;
        detection::obb::generate_proposals_yolov8_obb_native(grid_strides, feat_ptr, prob_threshold, proposals, input_w,
                                                             input_h, cls_num);
        detection::obb::get_out_obb_bbox(proposals, objects, nms_threshold, input_h, input_w, mat.rows, mat.cols);
        // detection::obb::draw_objects_obb(mat, objects, OBB_CLASS_NAMES, "yolo11_obb_out", 1);
    }
}

int EngineWrapper::Post_Process(cv::Mat& mat, int& input_w, int& input_h, int& cls_num, int& point_num,
                                float& pron_threshold, float& nms_threshold, std::vector<detection::Object>& objects,
                                std::string& model_type)
{
    post_process(m_io_info, &m_io, mat, input_w, input_h, cls_num, point_num, pron_threshold, nms_threshold, objects,
                 model_type);
    return 0;
}