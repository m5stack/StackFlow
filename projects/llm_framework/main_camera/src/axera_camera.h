/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef AXERA_CAMERA_H
#define AXERA_CAMERA_H
#include "common_venc.h"
#include "common_vin.h"
#if __cplusplus
extern "C" {
#endif

typedef enum {
    SAMPLE_VIN_NONE                      = -1,
    SAMPLE_VIN_SINGLE_DUMMY              = 0,
    SAMPLE_VIN_SINGLE_OS04A10            = 1,
    SAMPLE_VIN_DOUBLE_OS04A10            = 2,
    SAMPLE_VIN_SINGLE_SC450AI            = 3,
    SAMPLE_VIN_DOUBLE_SC450AI            = 4,
    SAMPLE_VIN_DOUBLE_OS04A10_AND_BT656  = 5,
    SAMPLE_VIN_SINGLE_S5KJN1SQ03         = 6,
    SAMPLE_VIN_SINGLE_OS04A10_DCG_HDR    = 7,
    SAMPLE_VIN_SINGLE_OS04A10_DCG_VS_HDR = 8,
    SYS_CASE_SINGLE_DVP                  = 20,
    SYS_CASE_SINGLE_BT601                = 21,
    SYS_CASE_SINGLE_BT656                = 22,
    SYS_CASE_SINGLE_BT1120               = 23,
    SYS_CASE_SINGLE_LVDS                 = 24,
    SYS_CASE_SINGLE_OS04A10_ONLINE       = 25,
    SMARTSENS_SC850SL                    = 13,
    SAMPLE_VIN_SINGLE_SC850SL            = 26,
    SAMPLE_VIN_BUTT
} SAMPLE_VIN_CASE_E;

typedef struct {
    SAMPLE_VIN_CASE_E eSysCase;
    COMMON_VIN_MODE_E eSysMode;
    AX_SNS_HDR_MODE_E eHdrMode;
    SAMPLE_LOAD_RAW_NODE_E eLoadRawNode;
    AX_BOOL bAiispEnable;
    AX_S32 nDumpFrameNum;
} SAMPLE_VIN_PARAM_T;

typedef struct axera_config_t {
    SAMPLE_VIN_PARAM_T VinParam;
} axera_config_t;

/**
 * Open the axera_camera
 * @pdev_name Device node
 * Return value: NULL for failure
 */
camera_t* axera_camera_open(const char* pdev_name, int width, int height, int fps, void* config);

/**
 * Open the axera_camera from config
 * @pDevName Device node
 * Return value: 0 for success, -1 for failure
 */
int axera_camera_open_from(camera_t* camera);

/**
 * Close the axera_camera
 * Return value: 0 for success, -1 for failure
 */
int axera_camera_close(camera_t* camera);

void init_rtsp(AX_VENC_CHN_ATTR_T *stVencChnAttr);
void init_jpeg(AX_VENC_CHN_ATTR_T *stVencChnAttr);

#if __cplusplus
}
#endif
#endif