/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "camera.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/prctl.h>
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
#include "axera_camera.h"
#include "AXRtspWrapper.h"
#include "ax_venc_api.h"
#include "ax_global_type.h"
#include "common_isp.h"
#include "common_sys.h"
#include "common_cam.h"
#include "common_isp.h"
#include "common_nt.h"
#include "common_vin.h"
#include "ax_sys_api.h"
#include "ax_ivps_api.h"
#endif

#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)

#ifndef ALIGN_UP
#define ALIGN_UP(x, a) ((((x) + ((a) - 1)) / a) * a)
#endif

#define USER_OUTPUT_CHN 2
#define RTSP_OUTPUT_CHN 1

AX_MIPI_RX_ATTR_T gSc850slMipiAttr = {
    .ePhyMode        = AX_MIPI_PHY_TYPE_DPHY,
    .eLaneNum        = AX_MIPI_DATA_LANE_4,
    .nDataRate       = 80,
    .nDataLaneMap[0] = 0,
    .nDataLaneMap[1] = 1,
    .nDataLaneMap[2] = 3,
    .nDataLaneMap[3] = 4,
    .nClkLane[0]     = 2,
    .nClkLane[1]     = 5,
};

AX_SNS_ATTR_T gSc850slSnsAttr = {
    .nWidth             = 3840,
    .nHeight            = 2160,
    .fFrameRate         = 30,
    .eSnsMode           = AX_SNS_LINEAR_MODE,
    .eRawType           = AX_RT_RAW10,
    .eBayerPattern      = AX_BP_RGGB,
    .bTestPatternEnable = AX_FALSE,
    // .nSettingIndex = 12,
};

AX_SNS_CLK_ATTR_T gSc850slSnsClkAttr = {
    .nSnsClkIdx  = 0,
    .eSnsClkRate = AX_SNS_CLK_24M,
};

AX_VIN_DEV_ATTR_T gSc850slDevAttr = {
    .bImgDataEnable    = AX_TRUE,
    .bNonImgDataEnable = AX_FALSE,
    .eDevMode          = AX_VIN_DEV_ONLINE,
    .eSnsIntfType      = AX_SNS_INTF_TYPE_MIPI_RAW,
    .tDevImgRgn[0]     = {0, 0, 3840, 2160},
    .tDevImgRgn[1]     = {0, 0, 3840, 2160},
    .tDevImgRgn[2]     = {0, 0, 3840, 2160},
    .tDevImgRgn[3]     = {0, 0, 3840, 2160},

    /* When users transfer special data, they need to configure VC&DT for szImgVc/szImgDt/szInfoVc/szInfoDt */
    //.tMipiIntfAttr.szImgVc[0] = 0,
    //.tMipiIntfAttr.szImgVc[1] = 1,
    //.tMipiIntfAttr.szImgDt[0] = 43,
    //.tMipiIntfAttr.szImgDt[1] = 43,
    //.tMipiIntfAttr.szInfoVc[0] = 31,
    //.tMipiIntfAttr.szInfoVc[1] = 31,
    //.tMipiIntfAttr.szInfoDt[0] = 63,
    //.tMipiIntfAttr.szInfoDt[1] = 63,

    .ePixelFmt      = AX_FORMAT_BAYER_RAW_10BPP_PACKED,
    .eBayerPattern  = AX_BP_RGGB,
    .eSnsMode       = AX_SNS_LINEAR_MODE,
    .eSnsOutputMode = AX_SNS_NORMAL,
    .tCompressInfo  = {AX_COMPRESS_MODE_NONE, 0},
    .tFrameRateCtrl = {AX_INVALID_FRMRATE, AX_INVALID_FRMRATE},
};

AX_VIN_PIPE_ATTR_T gSc850slPipeAttr = {
    .ePipeWorkMode  = AX_VIN_PIPE_NORMAL_MODE1,
    .tPipeImgRgn    = {0, 0, 3840, 2160},
    .nWidthStride   = 3840,
    .eBayerPattern  = AX_BP_RGGB,
    .ePixelFmt      = AX_FORMAT_BAYER_RAW_10BPP_PACKED,
    .eSnsMode       = AX_SNS_LINEAR_MODE,
    .tCompressInfo  = {AX_COMPRESS_MODE_LOSSY, 0},
    .tNrAttr        = {{0, {AX_COMPRESS_MODE_LOSSLESS, 0}}, {0, {AX_COMPRESS_MODE_NONE, 0}}},
    .tFrameRateCtrl = {AX_INVALID_FRMRATE, AX_INVALID_FRMRATE},
};

AX_VIN_CHN_ATTR_T gSc850slChn0Attr = {
    .nWidth         = 3840,
    .nHeight        = 2160,
    .nWidthStride   = 3840,
    .eImgFormat     = AX_FORMAT_YUV420_SEMIPLANAR,
    .nDepth         = 1,
    .tCompressInfo  = {AX_COMPRESS_MODE_LOSSY, 4},
    .tFrameRateCtrl = {AX_INVALID_FRMRATE, AX_INVALID_FRMRATE},
};



struct axera_camera_index_t {
    char name[48];
    SAMPLE_VIN_CASE_E index;
} axera_camera_index[] = {{"axera_single_dummy", SAMPLE_VIN_SINGLE_DUMMY},
                          {"axera_single_os04a10", SAMPLE_VIN_SINGLE_OS04A10},
                          {"axera_double_os04a10", SAMPLE_VIN_DOUBLE_OS04A10},
                          {"axera_single_sc450ai", SAMPLE_VIN_SINGLE_SC450AI},
                          {"axera_double_sc450ai", SAMPLE_VIN_DOUBLE_SC450AI},
                          {"axera_double_os04a10_and_bt656", SAMPLE_VIN_DOUBLE_OS04A10_AND_BT656},
                          {"axera_single_s5kjn1sq03", SAMPLE_VIN_SINGLE_S5KJN1SQ03},
                          {"axera_single_os04a10_dcg_hdr", SAMPLE_VIN_SINGLE_OS04A10_DCG_HDR},
                          {"axera_single_os04a10_dcg_vs_hdr", SAMPLE_VIN_SINGLE_OS04A10_DCG_VS_HDR},
                          {"axera_single_dvp", SYS_CASE_SINGLE_DVP},
                          {"axera_single_bt601", SYS_CASE_SINGLE_BT601},
                          {"axera_single_bt656", SYS_CASE_SINGLE_BT656},
                          {"axera_single_bt1120", SYS_CASE_SINGLE_BT1120},
                          {"axera_single_lvds", SYS_CASE_SINGLE_LVDS},
                          {"axera_single_os04a10_online", SYS_CASE_SINGLE_OS04A10_ONLINE},
                          {"axera_single_sc850sl", SAMPLE_VIN_SINGLE_SC850SL}};



/* comm pool */
COMMON_SYS_POOL_CFG_T gtSysCommPoolSingleDummySdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_16BPP, 6},                              /*vin raw16 use */
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 4, AX_COMPRESS_MODE_LOSSY, 4}, /*vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolSingleDVPSdr[] = {
    {2560, 1440, 2560, AX_FORMAT_YUV420_SEMIPLANAR, 10}, /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolSingleBTSdr[] = {
    {720, 480, 720, AX_FORMAT_YUV422_INTERLEAVED_UYVY, 25 * 2}, /*vin yuv422 uyvy use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolSingleLVDSSdr[] = {
    {1600, 1200, 1600, AX_FORMAT_YUV420_SEMIPLANAR, 10}, /* vin nv21/nv21 use */
};

/* private pool */
COMMON_SYS_POOL_CFG_T gtPrivatePoolSingleDummySdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_16BPP, 10, AX_COMPRESS_MODE_LOSSY, 4},
};

COMMON_SYS_POOL_CFG_T gtSysPrivatePoolSingleDVPSdr[] = {
    {2560, 1440, 2560, AX_FORMAT_BAYER_RAW_10BPP, 25 * 2}, /*vin raw10 use */
};

COMMON_SYS_POOL_CFG_T gtSysPrivatePoolSingleBTSdr[] = {
    {720, 480, 720, AX_FORMAT_YUV422_INTERLEAVED_UYVY, 25 * 2}, /*vin yuv422 uyvy use */
};

COMMON_SYS_POOL_CFG_T gtSysPrivatePoolSingleLVDSSdr[] = {
    {1600, 1200, 1600, AX_FORMAT_BAYER_RAW_10BPP, 25 * 2}, /*vin raw10 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolSingleOs04a10Sdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 3, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 4},                            /* vin nv21/nv21 use */
    {1920, 1080, 1920, AX_FORMAT_YUV420_SEMIPLANAR, 3},                            /* vin nv21/nv21 use */
    {720, 576, 720, AX_FORMAT_YUV420_SEMIPLANAR, 3},                               /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolSingleOs04a10Sdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 12, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw16 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolDoubleOs04a10Bt656Sdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 7, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
    {720, 480, 720, AX_FORMAT_YUV422_SEMIPLANAR, 6, AX_COMPRESS_MODE_NONE, 0},     /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolDoubleOs04a10Bt656Sdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 7, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
    {720, 480, 720, AX_FORMAT_YUV422_SEMIPLANAR, 5, AX_COMPRESS_MODE_NONE, 0},          /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolDoubleOs04a10Bt656Hdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 7, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
    {720, 480, 720, AX_FORMAT_YUV422_SEMIPLANAR, 7, AX_COMPRESS_MODE_NONE, 0},     /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolDoubleOs04a10Bt656Hdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 13, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
    {720, 480, 720, AX_FORMAT_YUV422_SEMIPLANAR, 5, AX_COMPRESS_MODE_NONE, 0},           /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolDoubleOs04a10Sdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 6, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolDoubleOs04a10Sdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 14, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolDoubleOs04a10SdrOnly[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 8, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolDoubleOs04a10Hdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 6, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolDoubleOs04a10Hdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 7, AX_COMPRESS_MODE_NONE, 0},  /* vin raw10 use */
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 7, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolSingleOs450aiSdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 3, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolSingleOs450aiSdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 8, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};

/*************************************/
COMMON_SYS_POOL_CFG_T gtSysCommPoolSingleSc850SlSdr[] = {
    {3840, 2160, 3840, AX_FORMAT_YUV420_SEMIPLANAR, 8, AX_COMPRESS_MODE_LOSSY, 8}, /* vin nv21/nv21 use */
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 4},                            /* vin nv21/nv21 use */
    {1920, 1080, 1920, AX_FORMAT_YUV420_SEMIPLANAR, 3},                            /* vin nv21/nv21 use */
    {720, 576, 720, AX_FORMAT_YUV420_SEMIPLANAR, 3},                               /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolSingleSc850SlSdr[] = {
    {3840, 2160, 3840, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 8, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};
/*************************************/

COMMON_SYS_POOL_CFG_T gtSysCommPoolDoubleSc450aiSdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 6, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolDoubleSc450aiSdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 7, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolDoubleSc450aiHdr[] = {
    {2688, 1520, 2688, AX_FORMAT_YUV420_SEMIPLANAR, 6, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolDoubleSc450aiHdr[] = {
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 7, AX_COMPRESS_MODE_NONE, 0},  /* vin raw10 use */
    {2688, 1520, 2688, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 7, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};

COMMON_SYS_POOL_CFG_T gtSysCommPoolSingles5kjn1sq03Sdr[] = {
    {1920, 1080, 1920, AX_FORMAT_YUV420_SEMIPLANAR, 3, AX_COMPRESS_MODE_LOSSY, 4}, /* vin nv21/nv21 use */
};

COMMON_SYS_POOL_CFG_T gtPrivatePoolSingles5kjn1sq03Sdr[] = {
    {1920, 1080, 1920, AX_FORMAT_BAYER_RAW_10BPP_PACKED, 8, AX_COMPRESS_MODE_LOSSY, 4}, /* vin raw10 use */
};

static AX_VOID __cal_dump_pool(COMMON_SYS_POOL_CFG_T pool[], AX_SNS_HDR_MODE_E eHdrMode, AX_S32 nFrameNum)
{
    if (NULL == pool) {
        return;
    }
    if (nFrameNum > 0) {
        switch (eHdrMode) {
            case AX_SNS_LINEAR_MODE:
                pool[0].nBlkCnt += nFrameNum;
                break;

            case AX_SNS_HDR_2X_MODE:
                pool[0].nBlkCnt += nFrameNum * 2;
                break;

            case AX_SNS_HDR_3X_MODE:
                pool[0].nBlkCnt += nFrameNum * 3;
                break;

            case AX_SNS_HDR_4X_MODE:
                pool[0].nBlkCnt += nFrameNum * 4;
                break;

            default:
                pool[0].nBlkCnt += nFrameNum;
                break;
        }
    }
}

static AX_VOID __set_pipe_hdr_mode(AX_U32 *pHdrSel, AX_SNS_HDR_MODE_E eHdrMode)
{
    if (NULL == pHdrSel) {
        return;
    }

    switch (eHdrMode) {
        case AX_SNS_LINEAR_MODE:
            *pHdrSel = 0x1;
            break;

        case AX_SNS_HDR_2X_MODE:
            *pHdrSel = 0x1 | 0x2;
            break;

        case AX_SNS_HDR_3X_MODE:
            *pHdrSel = 0x1 | 0x2 | 0x4;
            break;

        case AX_SNS_HDR_4X_MODE:
            *pHdrSel = 0x1 | 0x2 | 0x4 | 0x8;
            break;

        default:
            *pHdrSel = 0x1;
            break;
    }
}

static AX_VOID __set_vin_attr(AX_CAMERA_T *pCam, SAMPLE_SNS_TYPE_E eSnsType, AX_SNS_HDR_MODE_E eHdrMode,
                              COMMON_VIN_MODE_E eSysMode, AX_BOOL bAiispEnable)
{
    pCam->eSnsType                              = eSnsType;
    pCam->tSnsAttr.eSnsMode                     = eHdrMode;
    pCam->tDevAttr.eSnsMode                     = eHdrMode;
    pCam->eHdrMode                              = eHdrMode;
    pCam->eSysMode                              = eSysMode;
    pCam->tPipeAttr[pCam->nPipeId].eSnsMode     = eHdrMode;
    pCam->tPipeAttr[pCam->nPipeId].bAiIspEnable = bAiispEnable;

    if (eHdrMode > AX_SNS_LINEAR_MODE) {
        pCam->tDevAttr.eSnsOutputMode = AX_SNS_DOL_HDR;
    }

    if (COMMON_VIN_TPG == eSysMode) {
        pCam->tDevAttr.eSnsIntfType = AX_SNS_INTF_TYPE_TPG;
    }

    if (COMMON_VIN_LOADRAW == eSysMode) {
        pCam->bEnableDev = AX_FALSE;
    } else {
        pCam->bEnableDev = AX_TRUE;
    }
    pCam->bChnEn[0]    = AX_TRUE;
    pCam->bRegisterSns = AX_TRUE;

    return;
}

AX_S32 CUSTOM_COMMON_VIN_GetSnsConfig(SAMPLE_SNS_TYPE_E eSnsType, AX_MIPI_RX_ATTR_T *ptMipiAttr,
                                      AX_SNS_ATTR_T *ptSnsAttr, AX_SNS_CLK_ATTR_T *ptSnsClkAttr,
                                      AX_VIN_DEV_ATTR_T *pDevAttr, AX_VIN_PIPE_ATTR_T *pPipeAttr,
                                      AX_VIN_CHN_ATTR_T *pChnAttr)
{
    if (eSnsType == SMARTSENS_SC850SL) {
        memcpy(ptMipiAttr, &gSc850slMipiAttr, sizeof(AX_MIPI_RX_ATTR_T));
        memcpy(ptSnsAttr, &gSc850slSnsAttr, sizeof(AX_SNS_ATTR_T));
        memcpy(ptSnsClkAttr, &gSc850slSnsClkAttr, sizeof(AX_SNS_CLK_ATTR_T));
        memcpy(pDevAttr, &gSc850slDevAttr, sizeof(AX_VIN_DEV_ATTR_T));
        memcpy(pPipeAttr, &gSc850slPipeAttr, sizeof(AX_VIN_PIPE_ATTR_T));
        memcpy(&pChnAttr[0], &gSc850slChn0Attr, sizeof(AX_VIN_CHN_ATTR_T));
        return 0;
    }
    return COMMON_VIN_GetSnsConfig(eSnsType, ptMipiAttr, ptSnsAttr, ptSnsClkAttr, pDevAttr, pPipeAttr, pChnAttr);
}

AX_SENSOR_REGISTER_FUNC_T *CUSTOM_COMMON_ISP_GetSnsObj(SAMPLE_SNS_TYPE_E eSnsType)
{
    if (eSnsType == SMARTSENS_SC850SL) {
        AX_SENSOR_REGISTER_FUNC_T *ptSnsHdl = NULL;
        void *handler                       = NULL;
        AX_CHAR *err                        = NULL;
        AX_U16 i                            = 0;
        AX_CHAR *pSnsPath                   = "/opt/lib/libsns_sc850sl.so";
        AX_CHAR *pObjName                   = "gSnssc850slObj";

        if ((NULL != pSnsPath) && (NULL != pObjName)) {
            handler = dlopen((void *)pSnsPath, RTLD_LAZY);
            if (NULL == handler) {
                COMM_ISP_PRT("open %s fail!---%s\n", pSnsPath, dlerror());
                return NULL;
            }
            ptSnsHdl = (AX_SENSOR_REGISTER_FUNC_T *)dlsym(handler, pObjName);
            err      = dlerror();
            if (NULL != err) {
                ptSnsHdl = NULL;
                COMM_ISP_PRT("dlsym %s fail!\n", pObjName);
            }
        } else {
            ptSnsHdl = NULL;
            COMM_ISP_PRT("not find eSnsType = %d\n", eSnsType);
        }

        return ptSnsHdl;
    }
    return COMMON_ISP_GetSnsObj(eSnsType);
}

static AX_U32 __sample_case_single_dummy(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                         SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_S32 i                            = 0;
    AX_CAMERA_T *pCam                   = NULL;
    COMMON_VIN_MODE_E eSysMode          = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode          = pVinParam->eHdrMode;
    SAMPLE_LOAD_RAW_NODE_E eLoadRawNode = pVinParam->eLoadRawNode;
    pCam                                = &pCamList[0];
    pCommonArgs->nCamCnt                = 1;

    for (i = 0; i < pCommonArgs->nCamCnt; i++) {
        pCam          = &pCamList[i];
        pCam->nPipeId = 0;
        CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                       &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);

        pCam->nDevId                  = 0;
        pCam->nRxDev                  = 0;
        pCam->tSnsClkAttr.nSnsClkIdx  = 0;
        pCam->tDevBindPipe.nNum       = 1;
        pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
        pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
        pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
        pCam->eLoadRawNode            = eLoadRawNode;
        pCam->eInputMode              = AX_INPUT_MODE_MIPI;
        __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
        __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
        for (AX_S32 j = 0; j < AX_VIN_MAX_PIPE_NUM; j++) {
            pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
            pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
            strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
        }
    }

    return 0;
}

static AX_U32 __sample_case_single_dvp(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType, SAMPLE_VIN_PARAM_T *pVinParam,
                                       COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam          = NULL;
    COMMON_VIN_MODE_E eSysMode = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode = pVinParam->eHdrMode;
    pCommonArgs->nCamCnt       = 1;
    pCam                       = &pCamList[0];
    pCam->nPipeId              = 0;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                  = 0;
    pCam->nRxDev                  = 0;
    pCam->tSnsClkAttr.nSnsClkIdx  = 0;
    pCam->tDevBindPipe.nNum       = 1;
    pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
    pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode              = AX_INPUT_MODE_DVP;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    for (AX_S32 j = 0; j < AX_VIN_MAX_PIPE_NUM; j++) {
        pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
        pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
        strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
    }

    return 0;
}

static AX_U32 __sample_case_single_bt656(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                         SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam          = NULL;
    COMMON_VIN_MODE_E eSysMode = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode = pVinParam->eHdrMode;
    pCommonArgs->nCamCnt       = 1;
    pCam                       = &pCamList[0];
    pCam->nPipeId              = 2;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                                 = 2;
    pCam->nRxDev                                 = 2;
    pCam->tSnsClkAttr.nSnsClkIdx                 = 0;
    pCam->tDevBindPipe.nNum                      = 1;
    pCam->tDevBindPipe.nPipeId[0]                = pCam->nPipeId;
    pCam->ptSnsHdl[pCam->nPipeId]                = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                               = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode                             = AX_INPUT_MODE_BT656;
    pCam->tPipeAttr[pCam->nPipeId].ePipeWorkMode = AX_VIN_PIPE_ISP_BYPASS_MODE;
    pCam->tPipeAttr[pCam->nPipeId].ePixelFmt     = AX_FORMAT_YUV420_SEMIPLANAR;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    pCam->bRegisterSns = AX_FALSE;

    return 0;
}

static AX_U32 __sample_case_single_bt1120(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                          SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam          = NULL;
    COMMON_VIN_MODE_E eSysMode = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode = pVinParam->eHdrMode;
    pCommonArgs->nCamCnt       = 1;
    pCam                       = &pCamList[0];
    pCam->nPipeId              = 0;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                                 = 2;
    pCam->nRxDev                                 = 2;
    pCam->tSnsClkAttr.nSnsClkIdx                 = 0;
    pCam->tDevBindPipe.nNum                      = 1;
    pCam->tDevBindPipe.nPipeId[0]                = pCam->nPipeId;
    pCam->ptSnsHdl[pCam->nPipeId]                = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                               = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode                             = AX_INPUT_MODE_BT1120;
    pCam->tPipeAttr[pCam->nPipeId].ePipeWorkMode = AX_VIN_PIPE_ISP_BYPASS_MODE;
    pCam->tPipeAttr[pCam->nPipeId].ePixelFmt     = AX_FORMAT_YUV420_SEMIPLANAR;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    pCam->bRegisterSns = AX_FALSE;

    return 0;
}

static AX_U32 __sample_case_single_lvds(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                        SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam          = NULL;
    COMMON_VIN_MODE_E eSysMode = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode = pVinParam->eHdrMode;
    pCommonArgs->nCamCnt       = 1;
    pCam                       = &pCamList[0];
    pCam->nPipeId              = 0;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                  = 0;
    pCam->nRxDev                  = 0;
    pCam->tSnsClkAttr.nSnsClkIdx  = 0;
    pCam->tDevBindPipe.nNum       = 1;
    pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
    pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode              = AX_INPUT_MODE_LVDS;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    for (AX_S32 j = 0; j < AX_VIN_MAX_PIPE_NUM; j++) {
        pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
        pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
        strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
    }

    return 0;
}

static AX_U32 __sample_case_single_os04a10(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                           SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam                   = NULL;
    COMMON_VIN_MODE_E eSysMode          = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode          = pVinParam->eHdrMode;
    AX_S32 j                            = 0;
    SAMPLE_LOAD_RAW_NODE_E eLoadRawNode = pVinParam->eLoadRawNode;
    pCommonArgs->nCamCnt                = 1;
    pCam                                = &pCamList[0];
    pCam->nPipeId                       = 0;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                  = 0;
    pCam->nRxDev                  = 0;
    pCam->tSnsClkAttr.nSnsClkIdx  = 0;
    pCam->tDevBindPipe.nNum       = 1;
    pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
    pCam->eLoadRawNode            = eLoadRawNode;
    pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode              = AX_INPUT_MODE_MIPI;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    for (j = 0; j < pCam->tDevBindPipe.nNum; j++) {
        pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
        pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
        strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
    }
    return 0;
}

static AX_U32 __sample_case_single_sc850sl(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                           SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam                   = NULL;
    COMMON_VIN_MODE_E eSysMode          = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode          = pVinParam->eHdrMode;
    AX_S32 j                            = 0;
    SAMPLE_LOAD_RAW_NODE_E eLoadRawNode = pVinParam->eLoadRawNode;
    pCommonArgs->nCamCnt                = 1;
    pCam                                = &pCamList[0];
    pCam->nPipeId                       = 0;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                  = 0;
    pCam->nRxDev                  = 0;
    pCam->tSnsClkAttr.nSnsClkIdx  = 0;
    pCam->tDevBindPipe.nNum       = 1;
    pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
    pCam->eLoadRawNode            = eLoadRawNode;
    pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode              = AX_INPUT_MODE_MIPI;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    for (j = 0; j < pCam->tDevBindPipe.nNum; j++) {
        pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
        pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
        if (pCam->tPipeInfo[j].bAiispEnable) {
            if (eHdrMode <= AX_SNS_LINEAR_MODE) {
                strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/sc850sl_sdr.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
            } else {
                strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/sc850sl_hdr_2x.bin",
                        sizeof(pCam->tPipeInfo[j].szBinPath));
            }
        } else {
            strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
        }
    }
    return 0;
}

static AX_U32 __sample_case_double_os04a10(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                           SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam          = NULL;
    COMMON_VIN_MODE_E eSysMode = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode = pVinParam->eHdrMode;
    AX_S32 j = 0, i = 0;
    pCommonArgs->nCamCnt = 2;

    for (i = 0; i < pCommonArgs->nCamCnt; i++) {
        pCam          = &pCamList[i];
        pCam->nNumber = i;
        pCam->nPipeId = i;
        CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                       &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);

        pCam->nDevId = i;
        if (i == 0) {
            pCam->nRxDev   = 0;
            pCam->nI2cAddr = 0x36;
        } else {
            pCam->nRxDev   = 1;
            pCam->nI2cAddr = 0x36;
        }
        pCam->tSnsClkAttr.nSnsClkIdx  = 0;
        pCam->tDevBindPipe.nNum       = 1;
        pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
        pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
        pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
        if (eHdrMode == AX_SNS_LINEAR_MODE)
            pCam->tSnsAttr.nSettingIndex = 33;
        else
            pCam->tSnsAttr.nSettingIndex = 34;
        pCam->tMipiAttr.eLaneNum = AX_MIPI_DATA_LANE_2;
        pCam->eInputMode         = AX_INPUT_MODE_MIPI;
        __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
        __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
        if (pCam->nPipeId != 0 && eHdrMode == AX_SNS_HDR_2X_MODE) {
            pCam->tPipeAttr[pCam->nPipeId].tCompressInfo.enCompressMode = AX_COMPRESS_MODE_NONE;
        }
        for (j = 0; j < pCam->tDevBindPipe.nNum; j++) {
            pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
            pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
            strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
        }
    }
    return 0;
}

static AX_U32 __sample_case_single_sc450ai(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                           SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam                   = NULL;
    COMMON_VIN_MODE_E eSysMode          = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode          = pVinParam->eHdrMode;
    AX_S32 j                            = 0;
    SAMPLE_LOAD_RAW_NODE_E eLoadRawNode = pVinParam->eLoadRawNode;
    pCommonArgs->nCamCnt                = 1;

    pCam          = &pCamList[0];
    pCam->nPipeId = 0;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                  = 0;
    pCam->nRxDev                  = 0;
    pCam->tSnsClkAttr.nSnsClkIdx  = 0;
    pCam->tDevBindPipe.nNum       = 1;
    pCam->eLoadRawNode            = eLoadRawNode;
    pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
    pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode              = AX_INPUT_MODE_MIPI;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    for (j = 0; j < pCam->tDevBindPipe.nNum; j++) {
        pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
        pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
        if (pCam->tPipeInfo[j].bAiispEnable) {
            if (eHdrMode <= AX_SNS_LINEAR_MODE) {
                strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/sc450ai_sdr_dual3dnr.bin",
                        sizeof(pCam->tPipeInfo[j].szBinPath));
            } else {
                strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/sc450ai_hdr_2x_ainr.bin",
                        sizeof(pCam->tPipeInfo[j].szBinPath));
            }
        } else {
            strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
        }
    }
    return 0;
}

static AX_U32 __sample_case_double_sc450ai(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                           SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam          = NULL;
    COMMON_VIN_MODE_E eSysMode = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode = pVinParam->eHdrMode;
    AX_S32 j = 0, i = 0;
    pCommonArgs->nCamCnt = 2;

    for (i = 0; i < pCommonArgs->nCamCnt; i++) {
        pCam          = &pCamList[i];
        pCam->nNumber = i;
        pCam->nPipeId = i;
        CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                       &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);

        pCam->nDevId = i;
        if (i == 0) {
            pCam->nRxDev   = 0;
            pCam->nI2cAddr = 0x30;
        } else {
            pCam->nRxDev   = 1;
            pCam->nI2cAddr = 0x30;
        }
        pCam->tSnsClkAttr.nSnsClkIdx  = 0;
        pCam->tDevBindPipe.nNum       = 1;
        pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
        pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
        pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
        if (eHdrMode == AX_SNS_LINEAR_MODE)
            pCam->tSnsAttr.nSettingIndex = 33;
        else
            pCam->tSnsAttr.nSettingIndex = 35;
        pCam->tMipiAttr.eLaneNum = AX_MIPI_DATA_LANE_2;
        pCam->eInputMode         = AX_INPUT_MODE_MIPI;
        __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
        __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
        if (pCam->nPipeId != 0 && eHdrMode == AX_SNS_HDR_2X_MODE) {
            pCam->tPipeAttr[pCam->nPipeId].tCompressInfo.enCompressMode = AX_COMPRESS_MODE_NONE;
        }
        for (j = 0; j < pCam->tDevBindPipe.nNum; j++) {
            pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
            pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
            if (pCam->tPipeInfo[j].bAiispEnable) {
                if (eHdrMode <= AX_SNS_LINEAR_MODE) {
                    strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/sc450ai_sdr_dual3dnr.bin",
                            sizeof(pCam->tPipeInfo[j].szBinPath));
                } else {
                    strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/sc450ai_hdr_2x_ainr.bin",
                            sizeof(pCam->tPipeInfo[j].szBinPath));
                }
            } else {
                strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
            }
        }
    }
    return 0;
}

static AX_U32 __sample_case_double_os04a10_and_bt656(AX_CAMERA_T *pCamList, SAMPLE_VIN_PARAM_T *pVinParam,
                                                     COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam          = NULL;
    COMMON_VIN_MODE_E eSysMode = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode = pVinParam->eHdrMode;
    SAMPLE_SNS_TYPE_E eSnsType;
    AX_S32 j = 0, i = 0;
    pCommonArgs->nCamCnt = 3;

    for (i = 0; i < pCommonArgs->nCamCnt; i++) {
        if (i < 2)
            eSnsType = OMNIVISION_OS04A10;
        else {
            eSnsType = SAMPLE_SNS_BT656;
            eHdrMode = AX_SNS_LINEAR_MODE;
        }
        pCam          = &pCamList[i];
        pCam->nNumber = i;
        pCam->nPipeId = i;
        CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                       &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);

        pCam->nDevId = i;
        pCam->nRxDev = i;
        if (i == 0) {
            pCam->nI2cAddr = 0x36;
        } else if (i == 1) {
            pCam->nI2cAddr = 0x36;
        }
        pCam->tSnsClkAttr.nSnsClkIdx  = 0;
        pCam->tDevBindPipe.nNum       = 1;
        pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
        pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
        pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
        if (eHdrMode == AX_SNS_LINEAR_MODE)
            pCam->tSnsAttr.nSettingIndex = 33;
        else
            pCam->tSnsAttr.nSettingIndex = 34;
        pCam->tMipiAttr.eLaneNum = AX_MIPI_DATA_LANE_2;
        pCam->eInputMode         = AX_INPUT_MODE_MIPI;
        if (i == 2) {
            pCam->eInputMode   = AX_INPUT_MODE_BT656;
            pCam->bRegisterSns = AX_FALSE;
        }
        __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
        __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
        for (j = 0; j < pCam->tDevBindPipe.nNum; j++) {
            pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
            pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
            if (pCam->tPipeInfo[j].bAiispEnable && i < 2) {
                if (eHdrMode <= AX_SNS_LINEAR_MODE) {
                    strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/os04a10_sdr_dual3dnr.bin",
                            sizeof(pCam->tPipeInfo[j].szBinPath));
                } else {
                    strncpy(pCam->tPipeInfo[j].szBinPath, "/opt/etc/os04a10_hdr_2x_ainr.bin",
                            sizeof(pCam->tPipeInfo[j].szBinPath));
                }
            } else {
                strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
            }
        }
    }
    return 0;
}

static AX_U32 __sample_case_single_s5kjn1sq03(AX_CAMERA_T *pCamList, SAMPLE_SNS_TYPE_E eSnsType,
                                              SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs)
{
    AX_CAMERA_T *pCam                   = NULL;
    COMMON_VIN_MODE_E eSysMode          = pVinParam->eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode          = pVinParam->eHdrMode;
    AX_S32 j                            = 0;
    SAMPLE_LOAD_RAW_NODE_E eLoadRawNode = pVinParam->eLoadRawNode;
    pCommonArgs->nCamCnt                = 1;
    pCam                                = &pCamList[0];
    pCam->nPipeId                       = 0;
    CUSTOM_COMMON_VIN_GetSnsConfig(eSnsType, &pCam->tMipiAttr, &pCam->tSnsAttr, &pCam->tSnsClkAttr, &pCam->tDevAttr,
                                   &pCam->tPipeAttr[pCam->nPipeId], pCam->tChnAttr);
    pCam->nDevId                  = 0;
    pCam->nRxDev                  = 0;
    pCam->tSnsClkAttr.nSnsClkIdx  = 0;
    pCam->tDevBindPipe.nNum       = 1;
    pCam->tDevBindPipe.nPipeId[0] = pCam->nPipeId;
    pCam->eLoadRawNode            = eLoadRawNode;
    pCam->ptSnsHdl[pCam->nPipeId] = CUSTOM_COMMON_ISP_GetSnsObj(eSnsType);
    pCam->eBusType                = COMMON_ISP_GetSnsBusType(eSnsType);
    pCam->eInputMode              = AX_INPUT_MODE_MIPI;
    __set_pipe_hdr_mode(&pCam->tDevBindPipe.nHDRSel[0], eHdrMode);
    __set_vin_attr(pCam, eSnsType, eHdrMode, eSysMode, pVinParam->bAiispEnable);
    for (j = 0; j < pCam->tDevBindPipe.nNum; j++) {
        pCam->tPipeInfo[j].ePipeMode    = SAMPLE_PIPE_MODE_VIDEO;
        pCam->tPipeInfo[j].bAiispEnable = pVinParam->bAiispEnable;
        strncpy(pCam->tPipeInfo[j].szBinPath, "null.bin", sizeof(pCam->tPipeInfo[j].szBinPath));
    }
    return 0;
}

static AX_U32 __sample_case_config(AX_CAMERA_T *gCams, SAMPLE_VIN_PARAM_T *pVinParam, COMMON_SYS_ARGS_T *pCommonArgs,
                                   COMMON_SYS_ARGS_T *pPrivArgs)
{
    AX_CAMERA_T *pCamList      = gCams;
    SAMPLE_SNS_TYPE_E eSnsType = OMNIVISION_OS04A10;

    COMM_ISP_PRT("eSysCase %d, eSysMode %d, eHdrMode %d, bAiispEnable %d\n", pVinParam->eSysCase, pVinParam->eSysMode,
                 pVinParam->eHdrMode, pVinParam->bAiispEnable);

    switch (pVinParam->eSysCase) {
        case SYS_CASE_SINGLE_DVP:
            eSnsType = SAMPLE_SNS_DVP;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleDVPSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleDVPSdr) / sizeof(gtSysCommPoolSingleDVPSdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleDVPSdr;

            /* private pool config */
            __cal_dump_pool(gtSysPrivatePoolSingleDVPSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtSysPrivatePoolSingleDVPSdr) / sizeof(gtSysPrivatePoolSingleDVPSdr[0]);
            pPrivArgs->pPoolCfg    = gtSysPrivatePoolSingleDVPSdr;

            /* cams config */
            __sample_case_single_dvp(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SYS_CASE_SINGLE_BT601:
            eSnsType = SAMPLE_SNS_BT601;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleBTSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleBTSdr) / sizeof(gtSysCommPoolSingleBTSdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleBTSdr;

            /* private pool config */
            __cal_dump_pool(gtSysPrivatePoolSingleBTSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtSysPrivatePoolSingleBTSdr) / sizeof(gtSysPrivatePoolSingleBTSdr[0]);
            pPrivArgs->pPoolCfg    = gtSysPrivatePoolSingleBTSdr;

            /* cams config */
            __sample_case_single_bt656(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SYS_CASE_SINGLE_BT656:
            eSnsType = SAMPLE_SNS_BT656;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleBTSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleBTSdr) / sizeof(gtSysCommPoolSingleBTSdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleBTSdr;

            /* private pool config */
            __cal_dump_pool(gtSysPrivatePoolSingleBTSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtSysPrivatePoolSingleBTSdr) / sizeof(gtSysPrivatePoolSingleBTSdr[0]);
            pPrivArgs->pPoolCfg    = gtSysPrivatePoolSingleBTSdr;

            /* cams config */
            __sample_case_single_bt656(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SYS_CASE_SINGLE_BT1120:
            eSnsType = SAMPLE_SNS_BT1120;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleBTSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleBTSdr) / sizeof(gtSysCommPoolSingleBTSdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleBTSdr;

            /* private pool config */
            __cal_dump_pool(gtSysPrivatePoolSingleBTSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtSysPrivatePoolSingleBTSdr) / sizeof(gtSysPrivatePoolSingleBTSdr[0]);
            pPrivArgs->pPoolCfg    = gtSysPrivatePoolSingleBTSdr;

            /* cams config */
            __sample_case_single_bt1120(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SYS_CASE_SINGLE_LVDS:
            eSnsType = SAMPLE_SNS_LVDS;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleLVDSSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleLVDSSdr) / sizeof(gtSysCommPoolSingleLVDSSdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleLVDSSdr;

            /* private pool config */
            __cal_dump_pool(gtSysPrivatePoolSingleLVDSSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtSysPrivatePoolSingleLVDSSdr) / sizeof(gtSysPrivatePoolSingleLVDSSdr[0]);
            pPrivArgs->pPoolCfg    = gtSysPrivatePoolSingleLVDSSdr;

            /* cams config */
            __sample_case_single_lvds(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_SINGLE_OS04A10:
            eSnsType = OMNIVISION_OS04A10;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleOs04a10Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleOs04a10Sdr) / sizeof(gtSysCommPoolSingleOs04a10Sdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleOs04a10Sdr;

            /* private pool config */
            __cal_dump_pool(gtPrivatePoolSingleOs04a10Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtPrivatePoolSingleOs04a10Sdr) / sizeof(gtPrivatePoolSingleOs04a10Sdr[0]);
            pPrivArgs->pPoolCfg    = gtPrivatePoolSingleOs04a10Sdr;

            /* cams config */
            __sample_case_single_os04a10(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_DOUBLE_OS04A10:
            eSnsType = OMNIVISION_OS04A10;
            /* comm pool config */
            if (AX_SNS_LINEAR_MODE == pVinParam->eHdrMode) {
                __cal_dump_pool(gtSysCommPoolDoubleOs04a10Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pCommonArgs->nPoolCfgCnt =
                    sizeof(gtSysCommPoolDoubleOs04a10Sdr) / sizeof(gtSysCommPoolDoubleOs04a10Sdr[0]);
                pCommonArgs->pPoolCfg = gtSysCommPoolDoubleOs04a10Sdr;
            } else {
                __cal_dump_pool(gtSysCommPoolDoubleOs04a10Hdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pCommonArgs->nPoolCfgCnt =
                    sizeof(gtSysCommPoolDoubleOs04a10Hdr) / sizeof(gtSysCommPoolDoubleOs04a10Hdr[0]);
                pCommonArgs->pPoolCfg = gtSysCommPoolDoubleOs04a10Hdr;
            }

            /* private pool config */
            if (AX_SNS_LINEAR_MODE == pVinParam->eHdrMode) {
                __cal_dump_pool(gtPrivatePoolDoubleOs04a10Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pPrivArgs->nPoolCfgCnt =
                    sizeof(gtPrivatePoolDoubleOs04a10Sdr) / sizeof(gtPrivatePoolDoubleOs04a10Sdr[0]);
                pPrivArgs->pPoolCfg = gtPrivatePoolDoubleOs04a10Sdr;
            } else if (AX_SNS_LINEAR_ONLY_MODE == pVinParam->eHdrMode) {
                __cal_dump_pool(gtPrivatePoolDoubleOs04a10SdrOnly, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pPrivArgs->nPoolCfgCnt =
                    sizeof(gtPrivatePoolDoubleOs04a10SdrOnly) / sizeof(gtPrivatePoolDoubleOs04a10SdrOnly[0]);
                pPrivArgs->pPoolCfg = gtPrivatePoolDoubleOs04a10SdrOnly;
            } else {
                __cal_dump_pool(gtPrivatePoolDoubleOs04a10Hdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pPrivArgs->nPoolCfgCnt =
                    sizeof(gtPrivatePoolDoubleOs04a10Hdr) / sizeof(gtPrivatePoolDoubleOs04a10Hdr[0]);
                pPrivArgs->pPoolCfg = gtPrivatePoolDoubleOs04a10Hdr;
            }

            /* cams config */
            __sample_case_double_os04a10(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_DOUBLE_SC450AI:
            eSnsType = SMARTSENS_SC450AI;
            /* comm pool config */
            if (AX_SNS_LINEAR_MODE == pVinParam->eHdrMode) {
                __cal_dump_pool(gtSysCommPoolDoubleSc450aiSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pCommonArgs->nPoolCfgCnt =
                    sizeof(gtSysCommPoolDoubleSc450aiSdr) / sizeof(gtSysCommPoolDoubleSc450aiSdr[0]);
                pCommonArgs->pPoolCfg = gtSysCommPoolDoubleSc450aiSdr;
            } else {
                __cal_dump_pool(gtSysCommPoolDoubleSc450aiHdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pCommonArgs->nPoolCfgCnt =
                    sizeof(gtSysCommPoolDoubleSc450aiHdr) / sizeof(gtSysCommPoolDoubleSc450aiHdr[0]);
                pCommonArgs->pPoolCfg = gtSysCommPoolDoubleSc450aiHdr;
            }

            /* private pool config */
            if (AX_SNS_LINEAR_MODE == pVinParam->eHdrMode) {
                __cal_dump_pool(gtPrivatePoolDoubleSc450aiSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pPrivArgs->nPoolCfgCnt =
                    sizeof(gtPrivatePoolDoubleSc450aiSdr) / sizeof(gtPrivatePoolDoubleSc450aiSdr[0]);
                pPrivArgs->pPoolCfg = gtPrivatePoolDoubleSc450aiSdr;
            } else {
                __cal_dump_pool(gtPrivatePoolDoubleSc450aiHdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pPrivArgs->nPoolCfgCnt =
                    sizeof(gtPrivatePoolDoubleSc450aiHdr) / sizeof(gtPrivatePoolDoubleSc450aiHdr[0]);
                pPrivArgs->pPoolCfg = gtPrivatePoolDoubleSc450aiHdr;
            }

            /* cams config */
            __sample_case_double_sc450ai(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_SINGLE_SC450AI:
            eSnsType = SMARTSENS_SC450AI;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleOs450aiSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleOs450aiSdr) / sizeof(gtSysCommPoolSingleOs450aiSdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleOs450aiSdr;

            /* private pool config */
            __cal_dump_pool(gtPrivatePoolSingleOs450aiSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtPrivatePoolSingleOs450aiSdr) / sizeof(gtPrivatePoolSingleOs450aiSdr[0]);
            pPrivArgs->pPoolCfg    = gtPrivatePoolSingleOs450aiSdr;

            /* cams config */
            __sample_case_single_sc450ai(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_SINGLE_SC850SL:
            eSnsType = SMARTSENS_SC850SL;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingleSc850SlSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleSc850SlSdr) / sizeof(gtSysCommPoolSingleSc850SlSdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleSc850SlSdr;

            /* private pool config */
            __cal_dump_pool(gtPrivatePoolSingleSc850SlSdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt = sizeof(gtPrivatePoolSingleSc850SlSdr) / sizeof(gtPrivatePoolSingleSc850SlSdr[0]);
            pPrivArgs->pPoolCfg    = gtPrivatePoolSingleSc850SlSdr;

            /* cams config */
            __sample_case_single_sc850sl(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_DOUBLE_OS04A10_AND_BT656:
            /* comm pool config */
            if (AX_SNS_LINEAR_MODE == pVinParam->eHdrMode) {
                __cal_dump_pool(gtSysCommPoolDoubleOs04a10Bt656Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pCommonArgs->nPoolCfgCnt =
                    sizeof(gtSysCommPoolDoubleOs04a10Bt656Sdr) / sizeof(gtSysCommPoolDoubleOs04a10Bt656Sdr[0]);
                pCommonArgs->pPoolCfg = gtSysCommPoolDoubleOs04a10Bt656Sdr;
            } else {
                __cal_dump_pool(gtSysCommPoolDoubleOs04a10Bt656Hdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pCommonArgs->nPoolCfgCnt =
                    sizeof(gtSysCommPoolDoubleOs04a10Bt656Hdr) / sizeof(gtSysCommPoolDoubleOs04a10Bt656Hdr[0]);
                pCommonArgs->pPoolCfg = gtSysCommPoolDoubleOs04a10Bt656Hdr;
            }

            /* private pool config */
            if (AX_SNS_LINEAR_MODE == pVinParam->eHdrMode) {
                __cal_dump_pool(gtPrivatePoolDoubleOs04a10Bt656Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pPrivArgs->nPoolCfgCnt =
                    sizeof(gtPrivatePoolDoubleOs04a10Bt656Sdr) / sizeof(gtPrivatePoolDoubleOs04a10Bt656Sdr[0]);
                pPrivArgs->pPoolCfg = gtPrivatePoolDoubleOs04a10Bt656Sdr;
            } else {
                __cal_dump_pool(gtPrivatePoolDoubleOs04a10Bt656Hdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
                pPrivArgs->nPoolCfgCnt =
                    sizeof(gtPrivatePoolDoubleOs04a10Bt656Hdr) / sizeof(gtPrivatePoolDoubleOs04a10Bt656Hdr[0]);
                pPrivArgs->pPoolCfg = gtPrivatePoolDoubleOs04a10Bt656Hdr;
            }

            /* cams config */
            __sample_case_double_os04a10_and_bt656(pCamList, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_SINGLE_S5KJN1SQ03:
            eSnsType = SAMSUNG_S5KJN1SQ03;
            /* comm pool config */
            __cal_dump_pool(gtSysCommPoolSingles5kjn1sq03Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pCommonArgs->nPoolCfgCnt =
                sizeof(gtSysCommPoolSingles5kjn1sq03Sdr) / sizeof(gtSysCommPoolSingles5kjn1sq03Sdr[0]);
            pCommonArgs->pPoolCfg = gtSysCommPoolSingles5kjn1sq03Sdr;

            /* private pool config */
            __cal_dump_pool(gtPrivatePoolSingles5kjn1sq03Sdr, pVinParam->eHdrMode, pVinParam->nDumpFrameNum);
            pPrivArgs->nPoolCfgCnt =
                sizeof(gtPrivatePoolSingles5kjn1sq03Sdr) / sizeof(gtPrivatePoolSingles5kjn1sq03Sdr[0]);
            pPrivArgs->pPoolCfg = gtPrivatePoolSingles5kjn1sq03Sdr;

            /* cams config */
            __sample_case_single_s5kjn1sq03(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
        case SAMPLE_VIN_SINGLE_DUMMY:
        default:
            eSnsType = SAMPLE_SNS_DUMMY;
            /* pool config */
            pCommonArgs->nPoolCfgCnt = sizeof(gtSysCommPoolSingleDummySdr) / sizeof(gtSysCommPoolSingleDummySdr[0]);
            pCommonArgs->pPoolCfg    = gtSysCommPoolSingleDummySdr;

            /* private pool config */
            pPrivArgs->nPoolCfgCnt = sizeof(gtPrivatePoolSingleDummySdr) / sizeof(gtPrivatePoolSingleDummySdr[0]);
            pPrivArgs->pPoolCfg    = gtPrivatePoolSingleDummySdr;

            /* cams config */
            __sample_case_single_dummy(pCamList, eSnsType, pVinParam, pCommonArgs);
            break;
    }

    return 0;
}

typedef enum {
    AX_SENSOR_NONT             = 0b00000,
    AX_SENSOR_NPU_ENABLE       = 0b00001,
    AX_SENSOR_CAM_ENABLE       = 0b00010,
    AX_SENSOR_CAM_POOL_ENABLE  = 0b00100,
    AX_SENSOR_CAM_OPEN         = 0b01000,
    AX_SENSOR_GET_FRAME_THREAD = 0b10000,
} AX_SENSOR_STATUS;

struct axera_camera_t {
    COMMON_SYS_ARGS_T tPrivArgs;
    COMMON_SYS_ARGS_T tCommonArgs;
    AX_CAMERA_T gCams;
    SAMPLE_VIN_PARAM_T VinParam;
    AX_IMG_INFO_T ax_img;
    AX_VIDEO_FRAME_T out_img;
    int Chn;
    AX_VENC_CHN_ATTR_T stVencChnAttr;
    AX_VENC_CHN_ATTR_T stJPEGVencChnAttr;
    AX_IVPS_PIPELINE_ATTR_T stPipelineAttr;
    AX_RTSP_HANDLE pRtspHandle;
    AX_RTSP_ATTR_T stRtspAttr[MAX_RTSP_MAX_CHANNEL_NUM];
    pthread_t venc_thread_id_;
    int venc_run_;
    int venc_jpeg_run_;
} axera_obj = {0};

static int camera_capture_callback_set(struct camera_t *camera, vcamera_frame_get pcallback)
{
    if (camera->state_ == CAMERA_SATTE_CAP) {
        SLOGW("Set capture callback failed");
        return -1;
    }
    camera->pcallback_ = pcallback;
    return 0;
}

static void *camera_capture_thread(void *param)
{
    int Ret          = -1;
    camera_t *camera = (camera_t *)param;
    struct v4l2_buffer EnQueueBuf;
    struct v4l2_buffer DeQueueBuf;

    SLOGI("Start capture");

    while (camera->state_ & AX_SENSOR_GET_FRAME_THREAD) {
        AX_S32 axRet = AX_VIN_GetYuvFrame(axera_obj.gCams.nPipeId, axera_obj.Chn, &axera_obj.ax_img, 500);
        if (axRet == 0) {
            // axera_obj.ax_img.tFrameInfo.stVFrame.u64VirAddr[0] =
            // (AX_U64)AX_POOL_GetBlockVirAddr(axera_obj.ax_img.tFrameInfo.stVFrame.u32BlkId[0]);
            // axera_obj.ax_img.tFrameInfo.stVFrame.u64PhyAddr[0] =
            // AX_POOL_Handle2PhysAddr(axera_obj.ax_img.tFrameInfo.stVFrame.u32BlkId[0]); AX_S32
            // AX_IVPS_CropResizeTdp(const AX_VIDEO_FRAME_T *ptSrc, AX_VIDEO_FRAME_T *ptDst,
            //     const AX_IVPS_CROP_RESIZE_ATTR_T *ptAttr);
            // AX_IVPS_CROP_RESIZE_ATTR_T tAttr = {0};
            // AX_IVPS_CropResizeTdp(&axera_obj.ax_img.tFrameInfo.stVFrame, &axera_obj.out_img, &tAttr);
            // AX_VIN_ReleaseYuvFrame(axera_obj.gCams.nPipeId, axera_obj.Chn, &axera_obj.ax_img);
            // camera->pcallback_((void *)axera_obj.out_img.u64VirAddr[0], axera_obj.out_img.u32Width,
            //                    axera_obj.out_img.u32Height, axera_obj.out_img.u32FrameSize, camera->ctx_);
        } else {
            SLOGI("get ax img error! code:0x%x", axRet);
            usleep(10 * 1000);
        }
    }

    SLOGI("Stop capture");

    return NULL;
}

static AX_VOID *IVPS_GetFrameThread(AX_VOID *pArg)
{
    AX_S32 ret = 0;
    AX_VIDEO_FRAME_T tSrcFrame;
    AX_BLK BlkId;
    struct camera_t *camera = (struct camera_t *)pArg;
    SLOGI("IVPS Grp: %d, Chn: %d", 0, USER_OUTPUT_CHN - 1);
    uint64_t totalGetStream = 0;
    while (camera->state_ & AX_SENSOR_GET_FRAME_THREAD) {
        // sleep(1);
        ret = AX_IVPS_GetChnFrame(0, USER_OUTPUT_CHN - 1, &tSrcFrame, 1000);
        if (ret) {
            /* reach EOF */
            SLOGI("[WARN] CHN[%d] is empty ret:0x%x\n", 2, ret);
            usleep(10000);
            continue;
        }

        // SLOGI(
        //     "AX_IVPS_GetChnFrame(%lld): Chn:%d, (%d x %d) Stride:%d, FrameSize:%d, Phy:%llx, UserData:%llx, PTS:%llx,
        //     " "BlockId:%x\n", tSrcFrame.u64SeqNum, USER_OUTPUT_CHN-1, tSrcFrame.u32Width, tSrcFrame.u32Height,
        //     tSrcFrame.u32PicStride[0], tSrcFrame.u32FrameSize, tSrcFrame.u64PhyAddr[0], tSrcFrame.u64UserData,
        //     tSrcFrame.u64PTS, tSrcFrame.u32BlkId[0]);
        void *pFrame = (void *)AX_POOL_GetBlockVirAddr(tSrcFrame.u32BlkId[0]);
        camera->pcallback_(pFrame, tSrcFrame.u32Width, tSrcFrame.u32Height,
                           tSrcFrame.u32Width * tSrcFrame.u32Height * 2, camera->ctx_);
        if ((totalGetStream % 60) == 0) {
            memset(pFrame, 128, tSrcFrame.u32Width * tSrcFrame.u32Height * 2);
        }
        ret = AX_IVPS_ReleaseChnFrame(0, USER_OUTPUT_CHN - 1, &tSrcFrame);
        if (ret) {
            SLOGI("AX_IVPS_ReleaseFrame fail, ret=0x%x", ret);
            usleep(10000);
            continue;
        }
        // SLOGI("AX_IVPS_GetFrame");
        totalGetStream++;
    }
    SLOGI("IVPS END");
    return NULL;
}

static int camera_capture_start(struct camera_t *camera)
{
    SLOGI("Start capture thread");
    if (!camera->pcallback_) {
        SLOGW("Capture callback not set, start faild");
        return -1;
    }
    if (!(camera->state_ & AX_SENSOR_CAM_OPEN)) {
        SLOGW("Camera not open, start faild");
        return -1;
    }
    camera->state_ |= AX_SENSOR_GET_FRAME_THREAD;
    pthread_create(&camera->capture_thread_id_, NULL, IVPS_GetFrameThread, camera);
    return 0;
}

static int camera_capture_stop(struct camera_t *camera)
{
    SLOGI("Stop capture thread");
    camera->state_ &= ~((int)AX_SENSOR_GET_FRAME_THREAD);
    pthread_join(camera->capture_thread_id_, NULL);
    SLOGI("Capture thread stop");
    return 0;
}

static void camera_set_ctx(struct camera_t *camera, void *ctx)
{
    camera->ctx_ = ctx;
}

/* venc get stream task */
static void *VencGetStreamProc(void *arg)
{
    AX_S32 s32Ret = -1;
    AX_VENC_RECV_PIC_PARAM_T stRecvParam;
    AX_VENC_STREAM_T stStream = {0};
    AX_S16 syncType           = -1;
    FILE *pStrm               = NULL;
    int totalGetStream        = 0;
    s32Ret                    = AX_VENC_StartRecvFrame(RTSP_OUTPUT_CHN - 1, &stRecvParam);
    if (AX_SUCCESS != s32Ret) {
        SLOGI("AX_VENC_StartRecvFrame failed, s32Ret:0x%x", s32Ret);
        return NULL;
    }
    while (axera_obj.venc_run_) {
        s32Ret = AX_VENC_GetStream(RTSP_OUTPUT_CHN - 1, &stStream, 1000);
        if (AX_SUCCESS == s32Ret) {
            totalGetStream++;
            /* Send to RTSP */
            AX_BOOL bIFrame = (AX_VENC_INTRA_FRAME == stStream.stPack.enCodingType) ? AX_TRUE : AX_FALSE;
            AX_Rtsp_SendNalu(axera_obj.pRtspHandle, 0, stStream.stPack.pu8Addr, stStream.stPack.u32Len,
                             stStream.stPack.u64PTS, bIFrame);
            // SLOGI("VencChn %d: u64PTS:%lld pu8Addr:%p u32Len:%d enCodingType:%d", 2, stStream.stPack.u64PTS,
            //       stStream.stPack.pu8Addr, stStream.stPack.u32Len, stStream.stPack.enCodingType);
            s32Ret = AX_VENC_ReleaseStream(RTSP_OUTPUT_CHN - 1, &stStream);
            if (AX_SUCCESS != s32Ret) {
                SLOGE("VencChn %d: AX_VENC_ReleaseStream failed!s32Ret:0x%x", 0, s32Ret);
                usleep(10000);
                continue;
            }
        } else if (AX_ERR_VENC_FLOW_END == s32Ret) {
            SLOGE("VencChn %d: AX_VENC_GetStream end flow,exit!", 0);
            usleep(10000);
            continue;
        }
    }
EXIT:
    s32Ret = AX_VENC_StopRecvFrame(RTSP_OUTPUT_CHN - 1);
    if (0 != s32Ret) {
        SLOGE("VencChn %d:AX_VENC_StopRecvFrame failed,s32Ret:0x%x", 0, s32Ret);
        return NULL;
    }
    SLOGI("VencChn %d: Total get %u encoded frames. getStream Exit!", 0, totalGetStream);
    return NULL;
}

static AX_S32 SAMPLE_VENC_Init()
{
    AX_S32 VencChn = 0, s32Ret = 0;
    AX_VENC_MOD_ATTR_T stModAttr = {
        .enVencType                     = AX_VENC_MULTI_ENCODER,
        .stModThdAttr.u32TotalThreadNum = 1,
        .stModThdAttr.bExplicitSched    = AX_FALSE,
    };
    s32Ret = AX_VENC_Init(&stModAttr);
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_VENC_Init failed, s32Ret:0x%x", s32Ret);
        return s32Ret;
    }
    AX_S32 ret = AX_VENC_CreateChn(VencChn, &axera_obj.stVencChnAttr);
    if (AX_SUCCESS != ret) {
        SLOGE("VencChn %d: AX_VENC_CreateChn failed, s32Ret:0x%x", VencChn, ret);
        return -1;
    }
    AX_MOD_INFO_T srcMod, dstMod;
    srcMod.enModId  = AX_ID_IVPS;
    srcMod.s32GrpId = 0;
    srcMod.s32ChnId = RTSP_OUTPUT_CHN - 1;
    dstMod.enModId  = AX_ID_VENC;
    dstMod.s32GrpId = 0;
    dstMod.s32ChnId = RTSP_OUTPUT_CHN - 1;
    AX_SYS_Link(&srcMod, &dstMod);
    axera_obj.stRtspAttr[0].nChannel            = 0;
    axera_obj.stRtspAttr[0].stVideoAttr.bEnable = AX_TRUE;
    axera_obj.stRtspAttr[0].stVideoAttr.ePt     = axera_obj.stVencChnAttr.stVencAttr.enType;
    AX_Rtsp_Init(&axera_obj.pRtspHandle, &axera_obj.stRtspAttr[0], 1, 0);
    AX_Rtsp_Start(axera_obj.pRtspHandle);
    pthread_create(&axera_obj.venc_thread_id_, NULL, VencGetStreamProc, NULL);
    SLOGI("RTSP pRtspHandle:%p nChnNum:%d", axera_obj.pRtspHandle, 1);
    return 0;
}

static AX_S32 SAMPLE_VENC_DeInit()
{
    AX_S32 VencChn = 0, s32Ret = 0, s32Retry = 5;

    if (axera_obj.stVencChnAttr.stVencAttr.enType == PT_PCMU) {
        return s32Ret;
    }
    axera_obj.venc_run_ = 0;
    pthread_join(axera_obj.venc_thread_id_, NULL);
    AX_Rtsp_Stop(axera_obj.pRtspHandle);
    AX_Rtsp_Deinit(axera_obj.pRtspHandle);
    AX_MOD_INFO_T srcMod, dstMod;
    srcMod.enModId  = AX_ID_IVPS;
    srcMod.s32GrpId = 0;
    srcMod.s32ChnId = RTSP_OUTPUT_CHN - 1;
    dstMod.enModId  = AX_ID_VENC;
    dstMod.s32GrpId = 0;
    dstMod.s32ChnId = RTSP_OUTPUT_CHN - 1;
    AX_SYS_UnLink(&srcMod, &dstMod);

    s32Retry = 5;
    do {
        s32Ret = AX_VENC_DestroyChn(VencChn);
        if (AX_ERR_VENC_BUSY == s32Ret) {
            SLOGE("VencChn %d:AX_VENC_DestroyChn return AX_ERR_VENC_BUSY,retry...", VencChn);
            --s32Retry;
            usleep(100 * 1000);
        } else {
            break;
        }
    } while (s32Retry >= 0);

    if (s32Retry == -1 || AX_SUCCESS != s32Ret) {
        SLOGE("VencChn %d: AX_VENC_DestroyChn failed, s32Retry=%d, s32Ret=0x%x\n", VencChn, s32Retry, s32Ret);
    }
    s32Ret = AX_VENC_Deinit();
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_VENC_Deinit failed, s32Ret=0x%x", s32Ret);
        return s32Ret;
    }
    return 0;
}

void init_rtsp(AX_VENC_CHN_ATTR_T *stVencChnAttr)
{
    axera_obj.stVencChnAttr = *stVencChnAttr;
    axera_obj.venc_run_     = 1;
}

void init_jpeg(AX_VENC_CHN_ATTR_T *stVencChnAttr)
{
    axera_obj.stJPEGVencChnAttr = *stVencChnAttr;
    axera_obj.venc_jpeg_run_    = 1;
}

static int SAMPLE_IVPS_Init(AX_S32 nGrpId, camera_t *camera)
{
    AX_S32 s32Ret                          = 0, nChn;
    AX_IVPS_GRP_ATTR_T stGrpAttr           = {0};
    AX_IVPS_PIPELINE_ATTR_T stPipelineAttr = {0};

    s32Ret = AX_IVPS_Init();
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_IVPS_Init failed,s32Ret:0x%x", s32Ret);
        return s32Ret;
    }

    stGrpAttr.nInFifoDepth = 2;
    stGrpAttr.ePipeline    = AX_IVPS_PIPELINE_DEFAULT;
    s32Ret                 = AX_IVPS_CreateGrp(nGrpId, &stGrpAttr);
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_IVPS_CreateGrp failed,nGrp %d,s32Ret:0x%x", nGrpId, s32Ret);
        return s32Ret;
    }
    stPipelineAttr.nOutChnNum = 3;
    // vin 输入
    stPipelineAttr.tFilter[0][0].bEngage = AX_TRUE;
    if (axera_obj.stVencChnAttr.stVencAttr.u32PicWidthSrc > 1920) {
        stPipelineAttr.tFilter[0][0].nDstPicWidth  = axera_obj.gCams.tChnAttr[0].nWidth;
        stPipelineAttr.tFilter[0][0].nDstPicHeight = axera_obj.gCams.tChnAttr[0].nHeight;
        stPipelineAttr.tFilter[0][0].nDstPicStride = axera_obj.gCams.tChnAttr[0].nWidthStride;
    } else {
        stPipelineAttr.tFilter[0][0].nDstPicWidth  = 1920;
        stPipelineAttr.tFilter[0][0].nDstPicHeight = 1080;
        stPipelineAttr.tFilter[0][0].nDstPicStride = 1920;
    }

    stPipelineAttr.tFilter[0][0].eDstPicFormat = AX_FORMAT_YUV420_SEMIPLANAR;
    stPipelineAttr.tFilter[0][0].eEngine       = AX_IVPS_ENGINE_VPP;
    // rtps输出 chn0
    stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].bEngage = AX_TRUE;
    if (axera_obj.stVencChnAttr.stVencAttr.enType != PT_PCMU) {
        stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].nDstPicWidth  = axera_obj.stVencChnAttr.stVencAttr.u32PicWidthSrc;
        stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].nDstPicHeight = axera_obj.stVencChnAttr.stVencAttr.u32PicHeightSrc;
    } else {
        stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].nDstPicWidth  = 1280;
        stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].nDstPicHeight = 720;
    }
    stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].nDstPicStride =
        ALIGN_UP(stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].nDstPicWidth, 16);
    stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].eDstPicFormat = AX_FORMAT_YUV420_SEMIPLANAR;
    stPipelineAttr.tFilter[RTSP_OUTPUT_CHN][0].eEngine       = AX_IVPS_ENGINE_VPP;

    // 设备获取 chn2
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].bEngage       = AX_TRUE;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].nDstPicWidth  = camera->width_;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].nDstPicHeight = camera->height_;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].nDstPicStride =
        ALIGN_UP(stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].nDstPicWidth, 16);
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].eDstPicFormat         = AX_FORMAT_YUV422_INTERLEAVED_YUYV;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].eEngine               = AX_IVPS_ENGINE_TDP;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].tTdpCfg.eRotation     = AX_IVPS_ROTATION_0;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].tAspectRatio.eMode    = AX_IVPS_ASPECT_RATIO_MANUAL;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].tAspectRatio.tRect.nX = 0;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].tAspectRatio.tRect.nY =
        (int)(stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].nDstPicHeight -
              (((stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].nDstPicWidth * 1.0f) /
                (stPipelineAttr.tFilter[0][0].nDstPicWidth * 1.0f)) *
               (stPipelineAttr.tFilter[0][0].nDstPicHeight * 1.0f)));
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].tAspectRatio.tRect.nW = camera->width_;
    stPipelineAttr.tFilter[USER_OUTPUT_CHN][0].tAspectRatio.tRect.nH = camera->height_;
    stPipelineAttr.nOutFifoDepth[USER_OUTPUT_CHN - 1]                = 2;

    // // jpeg输出 chn1
    // memcpy(&stPipelineAttr.tFilter[2][0], &stPipelineAttr.tFilter[3][0], sizeof(AX_IVPS_FILTER_T));
    // stPipelineAttr.tFilter[2][0].eDstPicFormat      = AX_FORMAT_YUV420_SEMIPLANAR;

#ifdef SAMPLE_IVPS_CROPRESIZE_ENABLE
    AX_S32 nChnGetId                        = 0;
    stPipelineAttr.nOutFifoDepth[nChnGetId] = 1;
#endif
    s32Ret = AX_IVPS_SetPipelineAttr(nGrpId, &stPipelineAttr);
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_IVPS_SetPipelineAttr failed,nGrp %d,s32Ret:0x%x", nGrpId, s32Ret);
        return s32Ret;
    }
    for (nChn = 0; nChn < stPipelineAttr.nOutChnNum; nChn++) {
        s32Ret = AX_IVPS_EnableChn(nGrpId, nChn);
        if (AX_SUCCESS != s32Ret) {
            SLOGE("AX_IVPS_EnableChn failed,nGrp %d,nChn %d,s32Ret:0x%x", nGrpId, nChn, s32Ret);
            return s32Ret;
        }
    }
    s32Ret = AX_IVPS_StartGrp(nGrpId);
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_IVPS_StartGrp failed,nGrp %d,s32Ret:0x%x", nGrpId, s32Ret);
        return s32Ret;
    }
#ifdef SAMPLE_IVPS_CROPRESIZE_ENABLE
    s32Ret = IVPS_CropResizeThreadStart(nGrpId, nChnGetId);
    if (AX_SUCCESS != s32Ret) {
        SLOGE("IVPS_CropResizeThreadStart failed,nGrp %d,s32Ret:0x%x", nGrpId, s32Ret);
        return s32Ret;
    }
#endif
    AX_MOD_INFO_T srcMod, dstMod;
    srcMod.enModId  = AX_ID_VIN;
    srcMod.s32GrpId = 0;
    srcMod.s32ChnId = 0;

    dstMod.enModId  = AX_ID_IVPS;
    dstMod.s32GrpId = nGrpId;
    dstMod.s32ChnId = 0;
    AX_SYS_Link(&srcMod, &dstMod);
    return 0;
}

static AX_S32 SAMPLE_IVPS_DeInit(AX_S32 nGrpId)
{
    AX_S32 s32Ret = 0, nChn = 0;

    AX_MOD_INFO_T srcMod, dstMod;
    srcMod.enModId  = AX_ID_VIN;
    srcMod.s32GrpId = 0;
    srcMod.s32ChnId = 0;
    dstMod.enModId  = AX_ID_IVPS;
    dstMod.s32GrpId = nGrpId;
    dstMod.s32ChnId = 0;
    AX_SYS_UnLink(&srcMod, &dstMod);

#ifdef SAMPLE_IVPS_CROPRESIZE_ENABLE
    IVPS_CropResizeThreadStop();
#endif

    s32Ret = AX_IVPS_StopGrp(nGrpId);
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_IVPS_StopGrp failed,nGrp %d,s32Ret:0x%x", nGrpId, s32Ret);
        return s32Ret;
    }

    for (nChn = 0; nChn < 3; nChn++) {
        s32Ret = AX_IVPS_DisableChn(nGrpId, nChn);
        if (AX_SUCCESS != s32Ret) {
            SLOGE("AX_IVPS_DisableChn failed,nGrp %d,nChn %d,s32Ret:0x%x", nGrpId, nChn, s32Ret);
            return s32Ret;
        }
    }

    s32Ret = AX_IVPS_DestoryGrp(nGrpId);
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_IVPS_DestoryGrp failed,nGrp %d,s32Ret:0x%x", nGrpId, s32Ret);
        return s32Ret;
    }

    s32Ret = AX_IVPS_Deinit();
    if (AX_SUCCESS != s32Ret) {
        SLOGE("AX_IVPS_Deinit failed,s32Ret:0x%x", s32Ret);
        return s32Ret;
    }

    return 0;
}

int axera_camera_open_from(camera_t *camera)
{
    int Ret = -1;
    AX_S32 axRet;
    if (camera == NULL) return -1;
    axera_config_t *axera_config = (axera_config_t *)camera->custom_config_;
    /* Check whether the camera is already open or in an error state */
    SLOGI("Open camera %s...", camera->dev_name_);
    if (camera->state_ & AX_SENSOR_CAM_OPEN) {
        SLOGE("Error: camera was open or meet error, now state is: %d", camera->state_);
        goto ErrorHandle;
    }
    axera_obj.VinParam.eSysCase = SAMPLE_VIN_BUTT;
    for (int i = 0; i < sizeof(axera_camera_index) / sizeof(axera_camera_index[0]); i++) {
        if (strcmp(axera_camera_index[i].name, camera->dev_name_) == 0) {
            axera_obj.VinParam.eSysCase = axera_camera_index[i].index;
            break;
        }
    }
    if (axera_obj.VinParam.eSysCase == SAMPLE_VIN_BUTT) {
        SLOGE("Error: camera not support %s", camera->dev_name_);
        return -10;
    }

    axera_obj.VinParam.eSysMode     = axera_config->VinParam.eSysMode; // COMMON_VIN_SENSOR;
    axera_obj.VinParam.eHdrMode     = axera_config->VinParam.eHdrMode; // AX_SNS_LINEAR_MODE;
    axera_obj.VinParam.bAiispEnable = axera_config->VinParam.bAiispEnable; // AX_TRUE;
    // axera_obj.gCams.tChnAttr
    __sample_case_config(&axera_obj.gCams, &axera_obj.VinParam, &axera_obj.tCommonArgs, &axera_obj.tPrivArgs);
    COMMON_SYS_Init(&axera_obj.tCommonArgs);
    COMMON_NPU_Init();
    SAMPLE_IVPS_Init(0, camera);
    if (axera_obj.venc_run_) {
        SAMPLE_VENC_Init();
    }

    axRet = COMMON_CAM_Init();
    if (axRet) {
        COMM_ISP_PRT("COMMON_CAM_Init fail, ret:0x%x", axRet);
        return -1;
    }
    camera->state_ |= AX_SENSOR_CAM_ENABLE;
    axRet = COMMON_CAM_PrivPoolInit(&axera_obj.tPrivArgs);
    if (axRet) {
        COMM_ISP_PRT("COMMON_CAM_PrivPoolInit fail, ret:0x%x", axRet);
        return -2;
    }
    camera->state_ |= AX_SENSOR_CAM_POOL_ENABLE;
    /* Step5: Cam Open */
    axRet = COMMON_CAM_Open(&axera_obj.gCams, axera_obj.tCommonArgs.nCamCnt);
    if (axRet) {
        COMM_ISP_PRT("COMMON_CAM_Open fail, ret:0x%x", axRet);
        return -3;
    }
    axera_obj.Chn = AX_VIN_CHN_ID_MAIN;
    camera->state_ |= AX_SENSOR_CAM_OPEN;

    // axera_obj.out_img.u32Width        = camera->width_;
    // axera_obj.out_img.u32Height       = camera->height_;
    // axera_obj.out_img.u32PicStride[0] = ALIGN_UP(camera->width_, 16);
    // axera_obj.out_img.enImgFormat     = AX_FORMAT_YUV420_SEMIPLANAR;
    // axera_obj.out_img.u32FrameSize    = camera->width_ * camera->height_ * 3 / 2;
    // AX_SYS_MemAlloc(&axera_obj.out_img.u64PhyAddr[0], (AX_VOID **)&axera_obj.out_img.u64VirAddr[0],
    //                 ALIGN_UP(axera_obj.out_img.u32FrameSize, 0x100), 0x100, (AX_S8 *)"StackFlowCameraBuff");

    camera->camera_capture_callback_set = camera_capture_callback_set;
    camera->camera_capture_start        = camera_capture_start;
    camera->camera_capture_stop         = camera_capture_stop;
    camera->camera_set_ctx              = camera_set_ctx;
    camera->is_alloc_                   = 0;
    return 0;

ErrorHandle:
    SLOGE("Camera open meet error, now handle it");
    return -1;
}

camera_t *axera_camera_open(const char *pdev_name, int width, int height, int fps, void *config)
{
    int Ret          = -1;
    camera_t *camera = (camera_t *)malloc(sizeof(camera_t));
    if (camera == NULL) return NULL;
    memset(camera, 0, sizeof(camera_t));
    camera->buffer_cnt_ = CONFIG_CAPTURE_BUF_CNT;

    int CopyLen = strlen(pdev_name);
    if (CopyLen > CONFIG_DEVNAME_LEN - 1) {
        SLOGE("Error: device name length over limit: %d", CopyLen);
        goto ErrorHandle;
    }
    memset(camera->dev_name_, 0, CONFIG_DEVNAME_LEN);
    memcpy(camera->dev_name_, pdev_name, CopyLen);

    camera->width_         = width;
    camera->height_        = height;
    camera->capture_fps_   = fps;
    camera->custom_config_ = config;

    Ret = axera_camera_open_from(camera);
    if (Ret) {
        goto ErrorHandle;
    }
    camera->is_alloc_ = 1;
    SLOGI("camera %s open success", camera->dev_name_);
    return camera;

ErrorHandle:
    SLOGE("Camera open meet error, now handle it");
    free(camera);
    return NULL;
}

int axera_camera_close(camera_t *camera)
{
    if (camera == NULL) return -1;
    if (camera->state_ & AX_SENSOR_CAM_OPEN) {
        COMMON_CAM_Close(&axera_obj.gCams, axera_obj.tCommonArgs.nCamCnt);
        camera->state_ &= ~((int)AX_SENSOR_CAM_OPEN);
    }

    if (camera->state_ & AX_SENSOR_CAM_ENABLE) {
        COMMON_CAM_Deinit();
        camera->state_ &= ~((int)AX_SENSOR_CAM_ENABLE);
    }
    camera->state_ = AX_SENSOR_NONT;
    SAMPLE_IVPS_DeInit(0);
    SAMPLE_VENC_DeInit();
    COMMON_SYS_DeInit();
    if (camera->is_alloc_) free(camera);

    SLOGI("camera closed");

    return 0;
}
#endif