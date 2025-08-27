/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/
#include "EngineWrapper.hpp"
#include "utils/io.hpp"

#include <cstdlib>

#include <global_config.h>

#if defined(CONFIG_AX_650N_MSP_ENABLED)
static const char *strAlgoModelType[AX_ENGINE_MODEL_TYPE_BUTT] = {"3.6T", "7.2T", "18T"};
#endif

#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
static const char *strAlgoModelType[AX_ENGINE_MODEL_TYPE_BUTT] = {"HalfOCM", "FullOCM"};
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

#if defined(CONFIG_AX_650N_MSP_ENABLED)
static AX_S32 CheckModelVNpu(const std::string &strModel, const AX_ENGINE_MODEL_TYPE_T &eModelType,
                             const AX_S32 &nNpuType, AX_U32 &nNpuSet)
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
static AX_S32 CheckModelVNpu(const std::string &strModel, const AX_ENGINE_MODEL_TYPE_T &eModelType,
                             const AX_S32 &nNpuType, AX_U32 &nNpuSet)
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

int EngineWrapper::Init(const char *strModelPath, uint32_t nNpuType, uint32_t npuMode)
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
    AX_CHAR *pModelBufferVirAddr = nullptr;
    AX_U64 u64ModelBufferPhyAddr = 0;
    AX_U32 nModelBufferSize      = 0;

    std::vector<char> model_buffer;

    if (bLoadModelUseCmm) {
        if (!utils::read_file(strModelPath, (AX_VOID **)&pModelBufferVirAddr, u64ModelBufferPhyAddr,
                              nModelBufferSize)) {
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

int EngineWrapper::SetInput(void *pInput, int index)
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

int EngineWrapper::GetOutput(void *pOutput, int index)
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

void *EngineWrapper::GetOutputPtr(int index)
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

void post_process(AX_ENGINE_IO_INFO_T *io_info, AX_ENGINE_IO_T *io_data, const cv::Mat &mat, std::string &model_type,
                  std::string &byteString)
{
    if (model_type == "segment") {
        auto &output = io_data->pOutputs[0];
        auto &info   = io_info->pOutputs[0];
        cv::Mat feature(info.pShape[2], info.pShape[3], CV_32FC1, output.pVirAddr);

        double minVal, maxVal;
        cv::minMaxLoc(feature, &minVal, &maxVal);

        feature -= minVal;
        feature /= (maxVal - minVal);
        // feature = 1.f - feature;
        feature *= 255;

        feature.convertTo(feature, CV_8UC1);

        cv::Mat dst(info.pShape[2], info.pShape[3], CV_8UC3);
        cv::applyColorMap(feature, dst, cv::ColormapTypes::COLORMAP_MAGMA);
        cv::resize(dst, dst, cv::Size(mat.cols, mat.rows));

        std::vector<uchar> buf;
        cv::imencode(".jpg", dst, buf);
        byteString.assign(buf.begin(), buf.end());
    }
}

int EngineWrapper::Post_Process(cv::Mat &mat, std::string &model_type, std::string &byteString)
{
    post_process(m_io_info, &m_io, mat, model_type, byteString);
    return 0;
}