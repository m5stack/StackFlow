/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdarg.h>
#include "YuvHandler.h"
#include "ax_skel_api.h"
#include "ax_sys_api.h"
#include "ax_ivps_api.h"
#include "ax_venc_api.h"
#include "ax_buffer_tool.h"
#include "skel_log.h"
#include "ax_engine_api.h"
#include "frameMgr.h"
#include "attrParser.h"
#include "statMgr.h"
#include "common_vdec_api.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) ((((x) + ((align) - 1)) / (align)) * (align))
#endif

#ifndef ALIGN_DOWN
#define ALIGN_DOWN(x, align) (((x) / (align)) * (align))
#endif

#ifndef AX_MAX
#define AX_MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef AX_MIN
#define AX_MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define SKEL_SAMPLE_OUTPUT_BODY_PATH "body"
#define SKEL_SAMPLE_OUTPUT_VEHICLE_PATH "vehicle"
#define SKEL_SAMPLE_OUTPUT_CYCLE_PATH "cycle"
#define SKEL_SAMPLE_OUTPUT_FACE_PATH "face"
#define SKEL_SAMPLE_OUTPUT_PLATE_PATH "plate"
#define SKEL_SAMPLE_OUTPUT_LOG_FILE "output.txt"

#define SAMPLE_SKEL_SHIFT_LEFT_ALIGN(a) (1 << (a))
#define SAMPLE_SKEL_VDEC_WIDTH_ALIGN     SAMPLE_SKEL_SHIFT_LEFT_ALIGN(8)
#define SAMPLE_SKEL_HEIGHT_ALIGN     SAMPLE_SKEL_SHIFT_LEFT_ALIGN(6)

#define RUN_COMMAND(format, ...)    do {   \
                                        char cmd[512];   \
                                        snprintf(cmd, 512, format, ##__VA_ARGS__);  \
                                        system(cmd);    \
                                    } while(0);

AX_VOID LogSaveToFile(FILE *file, const char *fmt, ...) {
    if (file) {
        va_list args;
        char szLog[1024] = {0};

        va_start(args, fmt);

        AX_U32 len = vsnprintf(szLog, sizeof(szLog), (char *)fmt, args);
        if (len < (AX_U32)sizeof(szLog)) {
            szLog[len] = '\0';
        } else {
            szLog[sizeof(szLog) - 1] = '\0';
        }

        fwrite((AX_U8 *)szLog, 1, strlen(szLog), file);

        va_end(args);
    }
}
#define OUTPUT_LOG_SAVE(fmt, ...) LogSaveToFile(fpResultFile, fmt "\n", ##__VA_ARGS__)
#define BLACK_VIDEO_FRAME_COUNT 60
#define SKEL_SAMPLE_OBJECT_SIZE 512

struct skeleton {
    int connection[2];
    int left_right_neutral;
};

struct skeleton pairs[] = {{{15, 13}, 0}, {{13, 11}, 0}, {{16, 14}, 0}, {{14, 12}, 0}, {{11, 12}, 0},
                           {{5, 11}, 0},  {{6, 12}, 0},  {{5, 6}, 0},   {{5, 7}, 0},   {{6, 8}, 0},
                           {{7, 9}, 0},   {{8, 10}, 0},  {{1, 2}, 0},   {{0, 1}, 0},   {{0, 2}, 0},
                           {{1, 3}, 0},   {{2, 4}, 0},   {{0, 5}, 0},   {{0, 6}, 0}};

#define DETECT_SKEL_POINT_COUNT 256

typedef struct _AI_Detection_Box_t {
    AX_F32 fX, fY, fW, fH;
} AI_Detection_Box_t;

typedef struct _AI_Detection_Point_t {
    AX_F32 fX, fY;
} AI_Detection_Point_t;

typedef struct _AI_Detection_SkelResult_t {
    const AX_CHAR *pstrObjectCategory;
    AX_U8 nPointNum;
    AI_Detection_Point_t tPoint[DETECT_SKEL_POINT_COUNT];
    AI_Detection_Box_t tBox;
} AI_Detection_SkelResult_t;

static AX_U64 get_tick_count(AX_VOID) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

// static AX_U64 get_tick_count_us(AX_VOID) {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
// }

static AX_VOID ShowUsage(AX_VOID) {
    // printf("usage: ./%s <options> ...\n", SAMPLE_SKEL_NAME);
    printf("options:\n");
    printf("-i, \tInput File(yuv)\n");
    printf("-r, \tInput File Resolution(wxh)(yuv: should be input)\n");
    printf("-w, \tWrite result image to new jpg file((unsigned int), default=0)\n");
    printf("-o, \tSave result to file(file name)\n");
    printf("-m, \tModels deployment path(path name)\n");
    printf("-t, \tRepeat times((unsigned int), default=1)\n");
    printf("-I, \tInterval repeat time((unsigned int)ms, default=0)\n");
    printf("-c, \tConfidence((float: 0-1), default=0)\n");
    printf("-H, \tHuman track size limit((unsigned int), default=0)\n");
    printf("-V, \tVehicle track size limit((unsigned int), default=0)\n");
    printf("-C, \tCylcle track size limit((unsigned int), default=0)\n");
    printf("-d, \tSkel detect type((unsigned int), default=2)\n"
                "\t\t0: detect only\n"
                "\t\t1: detect + track\n"
                "\t\t2: detect + track + push\n");
    printf("-u, \tSkel push strategy((unsigned int), default=3)\n"
                "\t\t1: fast push strategy\n"
                "\t\t2: push strategy\n"
                "\t\t3: best push strategy\n");
    printf("-N, \tSkel NPU type((unsigned int), default=0(VNPU Disable)\n"
                "\t\t0: VNPU Disable\n"
                "\t\t1: STD-VNPU Default\n"
                "\t\t2: STD-VNPU1\n"
                "\t\t3: STD-VNPU2\n");
    printf("-p, \tSkel PPL((unsigned int), default=1)\n"
                "\t\t1: AX_SKEL_PPL_HVCP\n"
                "\t\t2: AX_SKEL_PPL_FACE\n");
    printf("-v, \tLog level((unsigned int), default=5)\n"
                "\t\t0: LOG_EMERGENCY_LEVEL\n"
                "\t\t1: LOG_ALERT_LEVEL\n"
                "\t\t2: LOG_CRITICAL_LEVEL\n"
                "\t\t3: LOG_ERROR_LEVEL\n"
                "\t\t4: LOG_WARN_LEVEL\n"
                "\t\t5: LOG_NOTICE_LEVEL\n"
                "\t\t6: LOG_INFO_LEVEL\n"
                "\t\t7: LOG_DEBUG_LEVEL\n");
    printf("-h, \tprint this message\n");
}

AX_S32 ParseConfigParam(const AX_SKEL_CONFIG_T *pstConfig) {
    if (pstConfig->nSize > 0 && pstConfig->pstItems) {
        for (size_t i = 0; i < pstConfig->nSize; i++) {
            if (pstConfig->pstItems[i].pstrType && pstConfig->pstItems[i].pstrValue) {
                // cmd: "body_max_target_count", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                if (strcmp(pstConfig->pstItems[i].pstrType, "body_max_target_count") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %d", pstConfig->pstItems[i].pstrType, (AX_U8)pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "vehicle_max_target_count", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "vehicle_max_target_count") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %d", pstConfig->pstItems[i].pstrType, (AX_U8)pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "cycle_max_target_count", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "cycle_max_target_count") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %d", pstConfig->pstItems[i].pstrType, (AX_U8)pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "body_confidence", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "body_confidence") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %f", pstConfig->pstItems[i].pstrType, pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "face_confidence", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "face_confidence") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %f", pstConfig->pstItems[i].pstrType, pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "vehicle_confidence", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "vehicle_confidence") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %f", pstConfig->pstItems[i].pstrType, pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "cycle_confidence", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "cycle_confidence") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %f", pstConfig->pstItems[i].pstrType, pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "plate_confidence", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "plate_confidence") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %f", pstConfig->pstItems[i].pstrType, pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "crop_encoder_qpLevel", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "crop_encoder_qpLevel") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf =
                            (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s: %f", pstConfig->pstItems[i].pstrType, pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "body_min_size",  value_type: AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "body_min_size") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T)) {
                        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *pstConf =
                            (AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s %dx%d", pstConfig->pstItems[i].pstrType, pstConf->nWidth, pstConf->nHeight);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "face_min_size",  value_type: AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "face_min_size") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T)) {
                        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *pstConf =
                            (AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s %dx%d", pstConfig->pstItems[i].pstrType, pstConf->nWidth, pstConf->nHeight);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "vehicle_min_size",  value_type: AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "vehicle_min_size") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T)) {
                        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *pstConf =
                            (AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s %dx%d", pstConfig->pstItems[i].pstrType, pstConf->nWidth, pstConf->nHeight);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "cycle_min_size",  value_type: AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "cycle_min_size") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T)) {
                        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *pstConf =
                            (AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s %dx%d", pstConfig->pstItems[i].pstrType, pstConf->nWidth, pstConf->nHeight);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "plate_min_size",  value_type: AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "plate_min_size") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T)) {
                        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *pstConf =
                            (AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s %dx%d", pstConfig->pstItems[i].pstrType, pstConf->nWidth, pstConf->nHeight);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
               // cmd: "detect_roi_polygon",  value_type: AX_SKEL_ROI_POLYGON_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "detect_roi_polygon") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_ROI_POLYGON_CONFIG_T)) {
                        AX_SKEL_ROI_POLYGON_CONFIG_T *pstConf = (AX_SKEL_ROI_POLYGON_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [%d]: nPointNum[%d]", pstConfig->pstItems[i].pstrType, pstConf->bEnable, pstConf->nPointNum);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_strategy",  value_type: AX_SKEL_PUSH_STRATEGY_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_strategy") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_PUSH_STRATEGY_T)) {
                        AX_SKEL_PUSH_STRATEGY_T *pstConf = (AX_SKEL_PUSH_STRATEGY_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [mode:%d, times:%d, count:%d, same:%d]", pstConfig->pstItems[i].pstrType, pstConf->ePushMode, pstConf->nIntervalTimes,
                                pstConf->nPushCounts, pstConf->bPushSameFrame);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "body_crop_encoder",  value_type: AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "body_crop_encoder") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *pstConf = (AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [%f, %f, %f, %f]", pstConfig->pstItems[i].pstrType,
                                pstConf->fScaleLeft, pstConf->fScaleRight,
                                pstConf->fScaleTop, pstConf->fScaleBottom);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "vehicle_crop_encoder",  value_type: AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "vehicle_crop_encoder") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *pstConf = (AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [%f, %f, %f, %f]", pstConfig->pstItems[i].pstrType,
                                pstConf->fScaleLeft, pstConf->fScaleRight,
                                pstConf->fScaleTop, pstConf->fScaleBottom);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "cycle_crop_encoder",  value_type: AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "cycle_crop_encoder") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *pstConf = (AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [%f, %f, %f, %f]", pstConfig->pstItems[i].pstrType,
                                pstConf->fScaleLeft, pstConf->fScaleRight,
                                pstConf->fScaleTop, pstConf->fScaleBottom);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "face_crop_encoder",  value_type: AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "face_crop_encoder") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *pstConf = (AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [%f, %f, %f, %f]", pstConfig->pstItems[i].pstrType,
                                pstConf->fScaleLeft, pstConf->fScaleRight,
                                pstConf->fScaleTop, pstConf->fScaleBottom);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "plate_crop_encoder",  value_type: AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "plate_crop_encoder") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *pstConf = (AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [%f, %f, %f, %f]", pstConfig->pstItems[i].pstrType,
                                pstConf->fScaleLeft, pstConf->fScaleRight,
                                pstConf->fScaleTop, pstConf->fScaleBottom);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_panorama",  value_type: AX_SKEL_PUSH_PANORAMA_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_panorama") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_PUSH_PANORAMA_CONFIG_T)) {
                        AX_SKEL_PUSH_PANORAMA_CONFIG_T *pstConf = (AX_SKEL_PUSH_PANORAMA_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [Enable: %d, Quality: %d]", pstConfig->pstItems[i].pstrType,
                                pstConf->bEnable, pstConf->nQuality);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_quality_body",  value_type: AX_SKEL_ATTR_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_quality_body") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T)) {
                        AX_SKEL_ATTR_FILTER_CONFIG_T *pstConf = (AX_SKEL_ATTR_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [Q: %f]", pstConfig->pstItems[i].pstrType, pstConf->stCommonAttrFilterConfig.fQuality);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_quality_vehicle",  value_type: AX_SKEL_ATTR_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_quality_vehicle") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T)) {
                        AX_SKEL_ATTR_FILTER_CONFIG_T *pstConf = (AX_SKEL_ATTR_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [Q: %f]", pstConfig->pstItems[i].pstrType, pstConf->stCommonAttrFilterConfig.fQuality);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_quality_cycle",  value_type: AX_SKEL_ATTR_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_quality_cycle") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T)) {
                        AX_SKEL_ATTR_FILTER_CONFIG_T *pstConf = (AX_SKEL_ATTR_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [Q: %f]", pstConfig->pstItems[i].pstrType, pstConf->stCommonAttrFilterConfig.fQuality);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_quality_face",  value_type: AX_SKEL_ATTR_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_quality_face") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T)) {
                        AX_SKEL_ATTR_FILTER_CONFIG_T *pstConf = (AX_SKEL_ATTR_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [W: %d, H: %d, P: %f, Y: %f, R: %f, B: %f]", pstConfig->pstItems[i].pstrType,
                                pstConf->stFaceAttrFilterConfig.nWidth, pstConf->stFaceAttrFilterConfig.nHeight,
                                pstConf->stFaceAttrFilterConfig.stPoseblur.fPitch, pstConf->stFaceAttrFilterConfig.stPoseblur.fYaw,
                                pstConf->stFaceAttrFilterConfig.stPoseblur.fRoll, pstConf->stFaceAttrFilterConfig.stPoseblur.fBlur);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_quality_plate",  value_type: AX_SKEL_ATTR_FILTER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_quality_plate") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T)) {
                        AX_SKEL_ATTR_FILTER_CONFIG_T *pstConf = (AX_SKEL_ATTR_FILTER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s [Q: %f]", pstConfig->pstItems[i].pstrType, pstConf->stCommonAttrFilterConfig.fQuality);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_bind_enable",  value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_bind_enable") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf = (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s : %f", pstConfig->pstItems[i].pstrType, pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                 // cmd: "track_enable",  value_type: AX_SKEL_COMMON_ENABLE_CONFIG_S *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "track_enable") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_ENABLE_CONFIG_T)) {
                        AX_SKEL_COMMON_ENABLE_CONFIG_T *pstConf = (AX_SKEL_COMMON_ENABLE_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s : %d", pstConfig->pstItems[i].pstrType, pstConf->bEnable);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_enable",  value_type: AX_SKEL_COMMON_ENABLE_CONFIG_S *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_enable") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_ENABLE_CONFIG_T)) {
                        AX_SKEL_COMMON_ENABLE_CONFIG_T *pstConf = (AX_SKEL_COMMON_ENABLE_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL get %s : %d", pstConfig->pstItems[i].pstrType, pstConf->bEnable);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "target_config",  value_type: AX_SKEL_TARGET_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "target_config") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_TARGET_CONFIG_T)) {
                        AX_SKEL_TARGET_CONFIG_T *pstConf = (AX_SKEL_TARGET_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        if (pstConf->pstItems) {
                            for (AX_U32 j = 0; j < pstConf->nSize; j ++) {
                                if (pstConf->pstItems[j].pstrObjectCategory) {
                                    ALOGI("SKEL set %s: [%d]%s", pstConfig->pstItems[i].pstrType, j, pstConf->pstItems[j].pstrObjectCategory);
                                }
                            }
                        }
                        else {
                            ALOGE("SKEL cmd: %s invalid param", pstConfig->pstItems[i].pstrType);
                        }
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_target_config",  value_type: AX_SKEL_TARGET_CONFIG_S *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_target_config") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_TARGET_CONFIG_T)) {
                        AX_SKEL_TARGET_CONFIG_T *pstConf = (AX_SKEL_TARGET_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        if (pstConf->pstItems) {
                            for (AX_U32 j = 0; j < pstConf->nSize; j ++) {
                                if (pstConf->pstItems[j].pstrObjectCategory) {
                                    ALOGI("SKEL set %s: [%d]%s", pstConfig->pstItems[i].pstrType, j, pstConf->pstItems[j].pstrObjectCategory);
                                }
                            }
                        }
                        else {
                            ALOGE("SKEL cmd: %s invalid param", pstConfig->pstItems[i].pstrType);
                        }
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "analyzer_attr_config",  value_type: AX_SKEL_ANALYZER_CONFIG_T *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "analyzer_attr_config") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_ANALYZER_CONFIG_T)) {
                        AX_SKEL_ANALYZER_CONFIG_T *pstConf = (AX_SKEL_ANALYZER_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        if (pstConf->peItems) {
                            for (AX_U32 j = 0; j < pstConf->nSize; j ++) {
                                ALOGI("SKEL set %s: [%d]%d", pstConfig->pstItems[i].pstrType, j, pstConf->peItems[j]);
                            }
                        }
                        else {
                            ALOGE("SKEL cmd: %s invalid param", pstConfig->pstItems[i].pstrType);
                        }
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "venc_attr_config", value_type: AX_SKEL_COMMON_THRESHOLD_CONFIG_S *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "venc_attr_config") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T)) {
                        AX_SKEL_COMMON_THRESHOLD_CONFIG_T *pstConf = (AX_SKEL_COMMON_THRESHOLD_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL set %s: %d", pstConfig->pstItems[i].pstrType, (AX_U32)pstConf->fValue);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                // cmd: "push_attr_always", value_type: AX_SKEL_COMMON_ENABLE_CONFIG_S *
                else if (strcmp(pstConfig->pstItems[i].pstrType, "push_attr_always") == 0) {
                    if (pstConfig->pstItems[i].nValueSize == sizeof(AX_SKEL_COMMON_ENABLE_CONFIG_T)) {
                        AX_SKEL_COMMON_ENABLE_CONFIG_T *pstConf = (AX_SKEL_COMMON_ENABLE_CONFIG_T *)pstConfig->pstItems[i].pstrValue;
                        ALOGI("SKEL set %s: %d", pstConfig->pstItems[i].pstrType, pstConf->bEnable);
                    } else {
                        ALOGE("SKEL %s size(%d) no match", pstConfig->pstItems[i].pstrType, pstConfig->pstItems[i].nValueSize);
                    }
                }
                else {
                    ALOGE("SKEL cmd: %s not support", pstConfig->pstItems[i].pstrType);
                }
            }
        }
    }

    return 0;
}

FILE* LoadFile(const AX_CHAR *pFile, AX_U64 *pLen)
{
    /* Reading input file */
    FILE *f_in = fopen(pFile, "rb");

    if (f_in) {
        if (pLen) {
            fseeko(f_in, 0L, SEEK_END);
            AX_U64 nFileSize = ftello(f_in);
            rewind(f_in);

            *pLen = (AX_U64)nFileSize;
        }
    }

    return f_in;
}

int ReLoadFile(FILE* pFile)
{
    if (pFile) {
        rewind(pFile);
        return 0;
    }

    return -1;
}

int NV12ToStrideNV12(AX_U64 nSrcPhyAddr,
                        AX_VOID *pSrcVirAddr,
                        AX_U32 nSrcSize,
                        AX_U32 nSrcStride,
                        AX_U32 nSrcHeight,
                        AX_U64 nDstPhyAddr,
                        AX_VOID *pDstVirAddr,
                        AX_U32 nDstSize,
                        AX_U32 nDstStride,
                        AX_U32 nDstHeight
                        ) {
    AX_U64 nStartTime = get_tick_count();

    // src check
    if (!pSrcVirAddr) {
        return -1;
    }

    if (nSrcSize != nSrcStride * nSrcHeight * 3 /2) {
        return -1;
    }

    // dst check
    if (!pDstVirAddr) {
        return -1;
    }

    if (nDstSize != nDstStride * nDstHeight * 3 /2) {
        return -1;
    }

    if (nDstSize < nSrcSize) {
        return -1;
    }

    if (nDstStride < nSrcStride) {
        return -1;
    }

    if (nDstHeight != nSrcHeight) {
        return -1;
    }

    AX_VOID *src = pSrcVirAddr;
    AX_VOID *dst = pDstVirAddr;
    for(AX_U32 i = 0; i < nSrcHeight * 3 / 2; i++) {
        memcpy(dst, src, nSrcStride);
        src += nSrcStride;
        dst += nDstStride;
    }

    AX_U64 nProcessElasped = get_tick_count() - nStartTime;

    ALOGD("NV12ToStrideNV12 %dx%d=>%dx%d elapse: %lld ms", nSrcStride, nSrcHeight, nDstStride, nDstHeight, nProcessElasped);

    return 0;
}

int LoadFileToMem(FILE *pFile, AX_U64 nPhyAddr, AX_VOID *pVirAddr, AX_S32 nSize)
{
    AX_S32 nReadSize = 0;

    if (pFile) {
        if (nPhyAddr != 0 && pVirAddr) {
            memset((AX_U8 *)pVirAddr, 0x00, nSize);
            nReadSize = fread((AX_U8 *)pVirAddr, 1, nSize, pFile);
        }
        else {
            fseeko(pFile, nSize, SEEK_CUR);
        }
    }

    return nReadSize;
}

int DecodeJpeg(FILE* InputFileHandle, AX_U64 OneYUVDataPhy, AX_VOID* OneYUVDataVir, AX_U32 nWidth, AX_U32 nHeight)
{
    SAMPLE_INPUT_FILE_INFO_T stStreamInfo;
    SAMPLE_STREAM_BUF_T stStreamBuf;
    size_t nReadLen = 0;
    AX_U64 streamPhyAddr = 0;
    AX_VOID *pStreamVirAddr = NULL;
    AX_VDEC_DEC_ONE_FRM_T decOneFrmParam;
    size_t nFileSize = 0;
    AX_U32 heightAlign = 0;
    AX_U32 frmStride = 0;
    frmStride = AX_COMM_ALIGN(nWidth * 8, AX_VDEC_WIDTH_ALIGN * 8) / 8;
    heightAlign = ALIGN_UP(nHeight, 16);
    AX_U32 nFrameSize = heightAlign * frmStride * 3 / 2;
    
    memset(&stStreamInfo, 0, sizeof(SAMPLE_INPUT_FILE_INFO_T));
    memset(&stStreamBuf, 0, sizeof(SAMPLE_STREAM_BUF_T));
    memset(&decOneFrmParam, 0, sizeof(AX_VDEC_DEC_ONE_FRM_T));

    rewind(InputFileHandle);
    fseek(InputFileHandle, 0, SEEK_END);
    nFileSize = (size_t)ftello(InputFileHandle);

    stStreamInfo.fInput = InputFileHandle;
    stStreamInfo.sFileSize = nFileSize;

    stStreamBuf.uBufSize = ((AX_U32)nFileSize) > STREAM_BUFFER_MAX_SIZE ? STREAM_BUFFER_MAX_SIZE : nFileSize;
    int ret = AX_SYS_MemAlloc(&streamPhyAddr, (AX_VOID **)&pStreamVirAddr,
                            stStreamBuf.uBufSize, 0x100, (AX_S8 *)"vdec_input_stream");
    if (ret != AX_SUCCESS) {
        ALOGE("AX_SYS_MemAlloc FAILED! uBufSize:0x%x ret:0x%x\n",
                        stStreamBuf.uBufSize, ret);
        return -1;
    }

    stStreamBuf.tBufAddr.pVirAddr = pStreamVirAddr;
    stStreamBuf.tBufAddr.u64PhyAddr = streamPhyAddr;

    ret = StreamParserReadFrameJpeg(&stStreamInfo, &stStreamBuf, &nReadLen);
    if (0 != ret) {
        ALOGE("StreamParserReadFrameJpeg failed! ret = 0x%x", ret);
        AX_SYS_MemFree(streamPhyAddr, pStreamVirAddr);
        return -1;
    }

    if (!nReadLen) {
        ALOGE("read jpeg frame FAILED!\n");
        AX_SYS_MemFree(streamPhyAddr, pStreamVirAddr);
        ret = AX_ERR_VDEC_STRM_ERROR;
        return ret;
    }

    decOneFrmParam.stStream.pu8Addr = (AX_U8*)stStreamBuf.tBufAddr.pVirAddr;
    decOneFrmParam.stStream.u64PhyAddr = stStreamBuf.tBufAddr.u64PhyAddr;
    decOneFrmParam.stStream.u32StreamPackLen = (AX_U32)nReadLen;

    decOneFrmParam.stFrame.u64VirAddr[0] = (AX_ULONG)OneYUVDataVir;
    decOneFrmParam.stFrame.u64VirAddr[1] = (AX_ULONG)OneYUVDataVir + frmStride * heightAlign;
    decOneFrmParam.stFrame.u64PhyAddr[0] = OneYUVDataPhy;
    decOneFrmParam.stFrame.u64PhyAddr[1] = OneYUVDataPhy + frmStride * heightAlign;
    decOneFrmParam.stFrame.u32FrameSize  = nFrameSize;

    ret = AX_VDEC_JpegDecodeOneFrame(&decOneFrmParam);
    if (ret != AX_SUCCESS) {
        ALOGE("AX_VDEC_JpegDecodeOneFrame FAILED! ret:0x%x %s\n",
                        ret, SampleVdecRetStr(ret));
        AX_SYS_MemFree(streamPhyAddr, pStreamVirAddr);
        return ret;
    }

    AX_SYS_MemFree(streamPhyAddr, pStreamVirAddr);

    return 0; 
}

int LoadFileToMemExt(const AX_CHAR *pFile, AX_U64 *pPhyAddr, AX_VOID **ppVirAddr, AX_U32 *pLen)
{
    /* Reading input file */
    FILE *f_in = fopen(pFile, "rb");
    if (f_in == NULL) {
        ALOGE("Unable to open input file\n");
        return -1;
    }

    /* file i/o pointer to full */
    fseek(f_in, 0L, SEEK_END);
    AX_U32 nFileSize = ftell(f_in);
    rewind(f_in);

    if (pPhyAddr && ppVirAddr) {
        AX_U64 nPhyAddr = 0;
        AX_VOID *pVirAddr = NULL;

        // pVirAddr = (AX_VOID *)malloc(nFileSize);
        AX_S32 nRet = AX_SYS_MemAlloc(&nPhyAddr, (AX_VOID **)&pVirAddr, nFileSize, 128, (AX_S8 *)"SKEL_TEST");

        if (nRet != 0) {
            fclose(f_in);
            ALOGE("AX_SYS_MemAlloc failed");
            return -1;
        }

        fread((AX_U8 *)pVirAddr, nFileSize, 1, f_in);

        *pPhyAddr = nPhyAddr;
        *ppVirAddr = pVirAddr;
    }

    if (pLen) {
        *pLen = nFileSize;
    }

    fclose(f_in);

    return 0;
}

int EncodeOneFrameToJpeg(const AX_CHAR *dstFile, AX_U32 nStride, AX_U32 nWidth, AX_U32 nHeight, AX_U64 nPhyAddr, AX_VOID *pVirAddr, AX_U32 nLen) {
    AX_S32 s32Ret = AX_SUCCESS;

    AX_U32 frameSize = nLen;
    AX_JPEG_ENCODE_ONCE_PARAMS_T stJpegEncodeOnceParam;
    memset(&stJpegEncodeOnceParam, 0, sizeof(stJpegEncodeOnceParam));

    AX_U64 outPhyAddr = 0;
    AX_VOID *outVirAddr = NULL;

    s32Ret = AX_SYS_MemAlloc(&outPhyAddr, &outVirAddr, frameSize, 128, (AX_S8 *)"SKEL-VENC");
    if (s32Ret) {
        ALOGE("alloc mem err, size(%d).", frameSize);
        return -1;
    }

    // output
    stJpegEncodeOnceParam.u32OutBufSize = frameSize;
    stJpegEncodeOnceParam.ulPhyAddr = outPhyAddr;
    stJpegEncodeOnceParam.pu8Addr = (AX_U8 *)outVirAddr;

    // input
    stJpegEncodeOnceParam.stJpegParam.u32Qfactor = 90;
    stJpegEncodeOnceParam.u32Width = nWidth;
    stJpegEncodeOnceParam.u32Height = nHeight;
    stJpegEncodeOnceParam.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;
    stJpegEncodeOnceParam.u32PicStride[0] = nStride;
    stJpegEncodeOnceParam.u32PicStride[1] = stJpegEncodeOnceParam.u32PicStride[0];
    stJpegEncodeOnceParam.u32PicStride[2] = 0;

    stJpegEncodeOnceParam.u64PhyAddr[0] = nPhyAddr;
    stJpegEncodeOnceParam.u64PhyAddr[1] = stJpegEncodeOnceParam.u64PhyAddr[0] + stJpegEncodeOnceParam.u32PicStride[0] * stJpegEncodeOnceParam.u32Height;
    stJpegEncodeOnceParam.u64PhyAddr[2] = 0;

    stJpegEncodeOnceParam.u64VirAddr[0] = (AX_ULONG)pVirAddr;
    stJpegEncodeOnceParam.u64VirAddr[1] = stJpegEncodeOnceParam.u64VirAddr[0] + stJpegEncodeOnceParam.u32PicStride[0] * stJpegEncodeOnceParam.u32Height;
    stJpegEncodeOnceParam.u64VirAddr[2] = 0;

    s32Ret = AX_VENC_JpegEncodeOneFrame(&stJpegEncodeOnceParam);
    if (AX_SUCCESS != s32Ret) {
        ALOGE("AX_VENC_JpegEncodeOneFrame fail, ret=0x%x", s32Ret);
        s32Ret = -1;
        goto JENC_EXIT;
    }

JENC_EXIT:
    if (0 == s32Ret && dstFile) {
        FILE *fp_w = fopen(dstFile, "wb");

        if (fp_w) {
            ALOGN("\tWrite new JPG result image to file: %s", dstFile);
            fwrite((AX_U8 *)stJpegEncodeOnceParam.pu8Addr, stJpegEncodeOnceParam.u32Len, 1, fp_w);
            fclose(fp_w);
        }
    }

    if (outPhyAddr && (NULL != outVirAddr)) {
        AX_SYS_MemFree(outPhyAddr, outVirAddr);
    }

    return s32Ret;
}

AX_BOOL FrameSkipCtrl(AX_S32 nSrcFrameRate, AX_S32 nDstFrameRate, AX_S32 nFrameSeqNum) {
    if (nFrameSeqNum < 1) {
        nFrameSeqNum = 1;
    }

    if (nSrcFrameRate == nDstFrameRate) {
        return AX_FALSE;
    }

    if (nDstFrameRate > nSrcFrameRate) {
        return AX_FALSE;
    }

    if ((nFrameSeqNum * nDstFrameRate / (nSrcFrameRate)) > ((nFrameSeqNum - 1) * nDstFrameRate / (nSrcFrameRate))) {
        return AX_FALSE;
    } else {
        return AX_TRUE;
    }
}

AX_S32 main(AX_S32 argc, AX_CHAR *argv[]) {
    AX_S32 nRet = 0;
    AX_S32 c;
    AX_S32 isExit = 0;
    AX_BOOL bJpgFile = AX_FALSE;
    FILE *InputFileHandle = NULL;
    const AX_CHAR *InputFile = NULL;
    const AX_CHAR *InputResolution = NULL;
    AX_S32 InputFileFrameCnt = 1;
    const AX_CHAR *ReWritePath = NULL;
    const AX_CHAR *SaveResultPath = NULL;
    const AX_CHAR *ModelsPath = NULL;
    FILE *fpResultFile = NULL;
    AX_S32 nRepeatTimes = 1;
    AX_S32 nPPL = AX_SKEL_PPL_HVCP;
    AX_S32 nDetectType = 2;
    AX_S32 nPushStrategy = 3;
    AX_S32 nNpuType = 0;
    AX_S32 nInterval = 0;
    AX_U32 nStride = 1920;
    AX_U32 nWidth = 1920;
    AX_U32 nHeight = 1080;
    AX_U32 nJencBufSize = 0;
    AX_U32 nFrameDepth = 1;
    AX_U32 nOneSize = 0;
    AX_U64 OneYUVDataPhy = 0;
    AX_VOID *OneYUVDataVir = NULL;
    AX_U32 nFrameSize = nStride * nHeight * 3 / 2;
    AX_U32 nCacheListDepth = 1;
    AX_S32 nSrcFrameRate = 25;
    AX_S32 nDstFrameRate = 25;
    AX_VOID *YUVDataVir = NULL;
    AX_U64 YUVDataPhy = 0;
    AX_VOID *YUVDataWrVir = NULL;
    AX_U64 YUVDataWrPhy = 0;
    AX_SKEL_HANDLE pHandle = NULL;
    AX_SKEL_FRAME_T stFrame = {0};
    AX_SKEL_RESULT_T *pstResult = NULL;
    AX_U64 nFileSize = 0;
    AX_U64 nStartTime = 0;
    AX_U64 nInitElasped = 0;
    AX_U64 nCreateElasped = 0;
    AX_U64 nProcessElasped = 0;
    AX_U64 nResultElasped = 0;
    AX_U64 nResultElaspedMin = (AX_U64)-1;
    AX_U64 nResultElaspedMax = 0;
    AX_U64 nResultElaspedTotal = 0;
    AX_F32 fConfidence = 0.0;
    // AX_U32 nHumantracksize = 0;
    // AX_U32 nVehicletracksize = 0;
    // AX_U32 nCycletracksize = 0;
    AX_U64 nSkelFrameId = 1;

#if defined(SAMPLE_SKEL_BUILD_VERSION)
    printf("SKEL sample: %s build: %s %s\n", SAMPLE_SKEL_BUILD_VERSION, __DATE__, __TIME__);
#endif

    while ((c = getopt(argc, argv, "i:r:I:w:o:m:t:d:u:p:j:v:c:N:H:V:C:h::")) != -1) {
        isExit = 0;
        switch (c) {
            case 'i':
                InputFile = (const AX_CHAR *)optarg;
                break;
            case 'r':
                InputResolution = (const AX_CHAR *)optarg;
                break;
            case 'I':
                nInterval = atoi(optarg);
                break;
            case 'w':
                ReWritePath = (const AX_CHAR *)optarg;
                break;
            case 'o':
                SaveResultPath = (const AX_CHAR *)optarg;
                break;
            case 'm':
                ModelsPath = (const AX_CHAR *)optarg;
                break;
            case 't':
                nRepeatTimes = atoi(optarg);
                break;
            case 'd':
                nDetectType = atoi(optarg);
                break;
            case 'u':
                nPushStrategy = atoi(optarg);
                break;
            case 'N':
                nNpuType = atoi(optarg);
                break;
            case 'p':
                nPPL = atoi(optarg);
                break;
            case 'j':
                nJencBufSize = atoi(optarg);
                break;
            case 'c':
                fConfidence = atof(optarg);
                break;
            // case 'H':
            //     nHumantracksize = atoi(optarg);
            //     break;
            // case 'V':
            //     nVehicletracksize = atoi(optarg);
            //     break;
            // case 'C':
            //     nCycletracksize = atoi(optarg);
            //     break;
            case 'v':
                log_level = atoi(optarg);
                break;
            case 'h':
                isExit = 1;
                break;
            case 'D':
                nCacheListDepth = atoi(optarg);
                break;
            default:
                isExit = 1;
                break;
        }
    }

    if (isExit || !InputFile || !InputResolution || (nPPL < AX_SKEL_PPL_HVCP) || (nPPL > AX_SKEL_PPL_MAX)
        || (log_level < 0 || log_level >= SKEL_LOG_MAX)
        || (fConfidence < 0 || fConfidence > 1)
        || (nDetectType < 0 || nDetectType > 2)
        || (nPushStrategy < 1 || nPushStrategy > 3)
        || (nNpuType > 3)) {
        ShowUsage();
        exit(0);
    }

    if (InputFile) {
        AX_U32 nInputFileLen = strlen(InputFile);

        if (nInputFileLen > 4 && (strcasecmp(&InputFile[nInputFileLen - 4], ".jpg") == 0)) {
            bJpgFile = AX_TRUE;
        }
    }

    if (nRepeatTimes <= 0) {
        nRepeatTimes = 1;
    }

    if (nInterval < 0) {
        nInterval = 0;
    }
    if (nCacheListDepth == 0) {
        nCacheListDepth = 1;
    }
    nInterval = nInterval * 1000;

    if (access(InputFile, 0) != 0) {
        ALOGE("%s not exist", InputFile);
        exit(0);
    }

    // clear output
    if (SaveResultPath) {
        RUN_COMMAND("rm -rf %s", SaveResultPath);
        RUN_COMMAND("mkdir -p %s", SaveResultPath);
        RUN_COMMAND("mkdir -p %s/%s", SaveResultPath, SKEL_SAMPLE_OUTPUT_BODY_PATH);
        RUN_COMMAND("mkdir -p %s/%s", SaveResultPath, SKEL_SAMPLE_OUTPUT_VEHICLE_PATH);
        RUN_COMMAND("mkdir -p %s/%s", SaveResultPath, SKEL_SAMPLE_OUTPUT_CYCLE_PATH);
        RUN_COMMAND("mkdir -p %s/%s", SaveResultPath, SKEL_SAMPLE_OUTPUT_FACE_PATH);
        RUN_COMMAND("mkdir -p %s/%s", SaveResultPath, SKEL_SAMPLE_OUTPUT_PLATE_PATH);
    }

    if (ReWritePath) {
        RUN_COMMAND("rm -rf %s", ReWritePath);
        RUN_COMMAND("mkdir -p %s", ReWritePath);
    }

    if (InputResolution) {
        AX_CHAR *temp_p = strstr(InputResolution, "x");

        if (!temp_p || strlen(temp_p) <= 1) {
            ShowUsage();
            exit(0);
        }

        nWidth = atoi(InputResolution);
        nStride = nWidth;
        nHeight = atoi(temp_p + 1);
    }

    nRet = AX_SYS_Init();
    if (0 != nRet) {
        ALOGE("AX_SYS_Init() fail, ret = 0x%x", nRet);
        exit(0);
    }

    nRet = AX_IVPS_Init();

    ALOGI("AX_IVPS_Init nRet = 0x%x", nRet);

    AX_VENC_MOD_ATTR_T stVencModAttr;
    memset(&stVencModAttr, 0x00, sizeof(stVencModAttr));
    stVencModAttr.enVencType = AX_VENC_MULTI_ENCODER;
    stVencModAttr.stModThdAttr.u32TotalThreadNum = 1;
    stVencModAttr.stModThdAttr.bExplicitSched = AX_FALSE;
    nRet = AX_VENC_Init(&stVencModAttr);
    if (AX_SUCCESS != nRet) {
        ALOGE("AX_VENC_Init FAILED! ret:0x%x\n", nRet);
        goto EXIT0;
    }

    ALOGI("AX_VENC_Init nRet = 0x%x", nRet);

    nRet = AX_VDEC_Init(NULL);
    if (AX_SUCCESS != nRet) {
        ALOGE("AX_VDEC_Init FAILED! ret:0x%x\n", nRet);
        goto EXIT0;
    }

    ALOGI("AX_VDEC_Init nRet = 0x%x", nRet);

    // init engine
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
#if defined(CONFIG_AX_650N_MSP_ENABLED)
    if (nNpuType == 0) {
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    }
    else if (nNpuType >= 1 && nNpuType <= 4) {
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_STD;
    }
    else if (nNpuType >= 5 && nNpuType <= 7) {
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;
    }
#elif defined(CHIP_AX620E)
    if (nNpuType == 0) {
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    }
    else {
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_ENABLE;
    }
#else
    // #error "NO CHIP SELECTED."
#endif

    nRet = AX_ENGINE_Init(&npu_attr);
    if (0 != nRet) {
        ALOGE("AX_NPU_SDK_EX_Init_with_attr() fail, ret = 0x%x", nRet);
        goto EXIT0;
    }

    nOneSize = nWidth * nHeight * 3 / 2;
    nFrameSize = nStride * nHeight * 3 / 2;

    InputFileHandle = LoadFile(InputFile, &nFileSize);

    if (bJpgFile) {
        InputFileFrameCnt = 1;
    }
    else {
        if (!InputFileHandle
        || (nFileSize % nOneSize) != 0) {
            ALOGE("%s file is not %dx%d", InputFile, nWidth, nHeight);
            goto EXIT1;
        }

        InputFileFrameCnt = nFileSize / nOneSize;
    }
    

    if (nWidth%2 == 1
        || nHeight%2 == 1) {
        ALOGE("wxh(%dx%d) should be even", nWidth, nHeight);
        goto EXIT1;
    }

    // YUVDataWrVir = (AX_VOID *)malloc(nFileSize);
    nRet = AX_SYS_MemAlloc(&YUVDataWrPhy, &YUVDataWrVir, nFrameSize, 128, (const AX_S8 *)"SKEL_TEST");

    if (!YUVDataWrVir) {
        ALOGE("malloc fail nRet=0x%x", nRet);
        goto EXIT1;
    }

    if (SaveResultPath) {
        AX_CHAR szPath[512] = {0};
        snprintf(szPath, 511, "%s/%s", SaveResultPath, SKEL_SAMPLE_OUTPUT_LOG_FILE);
        fpResultFile = fopen(szPath, "a+");

        if (!fpResultFile) {
            ALOGE("%s file not exist", szPath);
            goto EXIT1;
        }
    }

    nStartTime = get_tick_count();

    AX_SKEL_INIT_PARAM_T stInitParam = {0};

    if (ModelsPath) {
        stInitParam.pStrModelDeploymentPath = ModelsPath;
    }
    else {
        stInitParam.pStrModelDeploymentPath = "/opt/etc/skelModels";
    }

    nRet = AX_SKEL_Init(&stInitParam);

    nInitElasped = get_tick_count() - nStartTime;

    if (0 != nRet) {
        ALOGE("SKEL init fail, ret = 0x%x", nRet);

        goto EXIT1;
    }

    // get version
    {
        const AX_SKEL_VERSION_INFO_T *pstVersion = NULL;

        nRet = AX_SKEL_GetVersion(&pstVersion);

        if (0 != nRet) {
            ALOGI("SKEL get version fail, ret = 0x%x", nRet);

            if (pstVersion) {
                AX_SKEL_Release((AX_VOID *)pstVersion);
            }
            goto EXIT1;
        }

        ALOGI("SKEL version: %s", pstVersion->pstrVersion);

        if (pstVersion) {
            AX_SKEL_Release((AX_VOID *)pstVersion);
        }
    }

    // get capability
    {
        const AX_SKEL_CAPABILITY_T *pstCapability = NULL;

        nRet = AX_SKEL_GetCapability(&pstCapability);

        if (0 != nRet) {
            ALOGI("SKEL get capability fail, ret = 0x%x", nRet);

            if (pstCapability) {
                AX_SKEL_Release((AX_VOID *)pstCapability);
            }
            goto EXIT1;
        }

        for (AX_U32 i = 0; i < pstCapability->nPPLConfigSize; i++) {
            ALOGI("SKEL capability[%d]: (ePPL: %d, PPLConfigKey: %s)", i, pstCapability->pstPPLConfig[i].ePPL,
                  pstCapability->pstPPLConfig[i].pstrPPLConfigKey);
        }

        if (pstCapability) {
            AX_SKEL_Release((AX_VOID *)pstCapability);
        }
    }

    AX_SKEL_HANDLE_PARAM_T stHandleParam = {0};

    stHandleParam.ePPL = (AX_SKEL_PPL_E)nPPL;
    stHandleParam.nFrameDepth = nFrameDepth;
    stHandleParam.nFrameCacheDepth = nCacheListDepth;
    stHandleParam.nIoDepth = 0;
    stHandleParam.nWidth = nWidth;
    stHandleParam.nHeight = nHeight;

    // config settings (if need)
    AX_SKEL_CONFIG_T stConfig = {0};
    AX_SKEL_CONFIG_ITEM_T stItems[16] = {0};
    AX_U8 itemIndex = 0;
    stConfig.nSize = 0;
    stConfig.pstItems = &stItems[0];

    // venc_attr_config
    // default will be system definition: w*h*3/8
    AX_SKEL_COMMON_THRESHOLD_CONFIG_T stVencAttrConfigThreshold = {0};
    if (nJencBufSize > 0) {
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"venc_attr_config";
        stVencAttrConfigThreshold.fValue = (AX_F32) nJencBufSize;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stVencAttrConfigThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;
    }

    AX_SKEL_COMMON_THRESHOLD_CONFIG_T stTrackEnableThreshold = {0};
    AX_SKEL_COMMON_THRESHOLD_CONFIG_T stPushEnableThreshold = {0};

    // detect only (disable track + disable push)
    if (nDetectType == 0) {
        // track_disable
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"track_enable";
        stTrackEnableThreshold.fValue = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stTrackEnableThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;

        // push_disable
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_enable";
        stPushEnableThreshold.fValue = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPushEnableThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;
    }
    // detect + track (disable push)
    else if (nDetectType == 1) {
        // push_disable
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_enable";
        stPushEnableThreshold.fValue = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPushEnableThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;
    }

    if (itemIndex > 0) {
        stConfig.nSize = itemIndex;
        stHandleParam.stConfig = stConfig;
    }

    // default NPU
    if (nNpuType == 0 || nNpuType == 1) {
        stHandleParam.nNpuType = (AX_S32)AX_SKEL_NPU_DEFAULT;
    }
    // STD-NPU
    else if (nNpuType == 2) {
        stHandleParam.nNpuType = (AX_S32)AX_SKEL_STD_VNPU_1;
    }
    else if (nNpuType == 3) {
        stHandleParam.nNpuType = (AX_S32)AX_SKEL_STD_VNPU_2;
    }

    nStartTime = get_tick_count();

    nRet = AX_SKEL_Create(&stHandleParam, &pHandle);

    nCreateElasped = get_tick_count() - nStartTime;

    if (0 != nRet) {
        ALOGE("SKEL Create Handle fail, ret = 0x%x", nRet);

        goto EXIT2;
    }

    if (fConfidence == 0) {
        fConfidence = 0.5;
    }

    // set config
    {
        AX_SKEL_CONFIG_T stConfig = {0};
        AX_SKEL_CONFIG_ITEM_T stItems[64] = {0};
        AX_U8 itemIndex = 0;
        stConfig.nSize = 0;
        stConfig.pstItems = &stItems[0];
#if 0
        // body_max_target_count
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"body_max_target_count";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stBodyMaxTargetCount = {0};
        stBodyMaxTargetCount.fValue = (AX_F32)nHumantracksize;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stBodyMaxTargetCount;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;

        // vehicle_max_target_count
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"vehicle_max_target_count";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stVehicleMaxTargetCount = {0};
        stVehicleMaxTargetCount.fValue = (AX_F32)nVehicletracksize;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stVehicleMaxTargetCount;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;

        // cycle_max_target_count
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"cycle_max_target_count";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stCycleMaxTargetCount = {0};
        stCycleMaxTargetCount.fValue = (AX_F32)nCycletracksize;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stCycleMaxTargetCount;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;
#endif
#if 0
        // body_confidence
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"body_confidence";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stBodyConfidence = {0};
        stBodyConfidence.fValue = fConfidence;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stBodyConfidence;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;

        // face_confidence
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"face_confidence";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stFaceConfidence = {0};
        stFaceConfidence.fValue = fConfidence;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stFaceConfidence;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;

        // vehicle_confidence
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"vehicle_confidence";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stVehicleConfidence = {0};
        stVehicleConfidence.fValue = fConfidence;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stVehicleConfidence;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;

        // cycle_confidence
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"cycle_confidence";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stCycleConfidence = {0};
        stCycleConfidence.fValue = fConfidence;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stCycleConfidence;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;

        // plate_confidence
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"plate_confidence";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stPlateConfidence = {0};
        stPlateConfidence.fValue = fConfidence;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPlateConfidence;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;
#endif

#if 0
        // crop_encoder_qpLevel
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"crop_encoder_qpLevel";
        AX_SKEL_COMMON_THRESHOLD_CONFIG_T stCropEncoderQpLevelThreshold = {0};
        stCropEncoderQpLevelThreshold.fValue = 90;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stCropEncoderQpLevelThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_THRESHOLD_CONFIG_T);
        itemIndex++;
#endif

#if 0
        // body_min_size
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"body_min_size";
        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T stBodyMinSize = {0};
        stBodyMinSize.nWidth = 0;
        stBodyMinSize.nHeight = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stBodyMinSize;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T);
        itemIndex++;

        // face_min_size
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"face_min_size";
        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T stFaceMinSize = {0};
        stFaceMinSize.nWidth = 0;
        stFaceMinSize.nHeight = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stFaceMinSize;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T);
        itemIndex++;

        // vehicle_min_size
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"vehicle_min_size";
        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T stVehicleMinSize = {0};
        stVehicleMinSize.nWidth = 0;
        stVehicleMinSize.nHeight = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stVehicleMinSize;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T);
        itemIndex++;

        // cycle_min_size
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"cycle_min_size";
        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T stCycleMinSize = {0};
        stCycleMinSize.nWidth = 0;
        stCycleMinSize.nHeight = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stCycleMinSize;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T);
        itemIndex++;

        // plate_min_size
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"plate_min_size";
        AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T stPlateMinSize = {0};
        stPlateMinSize.nWidth = 0;
        stPlateMinSize.nHeight = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPlateMinSize;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_OBJECT_SIZE_FILTER_CONFIG_T);
        itemIndex++;
#endif

        // detect_roi_polygon
        AX_SKEL_ROI_POLYGON_CONFIG_T stDetectRoi = {0};
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"detect_roi_polygon";
        stDetectRoi.bEnable = AX_FALSE;
        stDetectRoi.nPointNum = 4;
        AX_SKEL_POINT_T stPoint[4] = {{0, 0}, {nWidth, 0}, {nWidth, nHeight}, {0, nHeight}};
        stDetectRoi.pstPoint = (AX_SKEL_POINT_T *)stPoint;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stDetectRoi;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_ROI_POLYGON_CONFIG_T);
        itemIndex++;

        // push_strategy
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_strategy";
        AX_SKEL_PUSH_STRATEGY_T stPushStrategy = {0};
        stPushStrategy.ePushMode = (AX_SKEL_PUSH_MODE_E)nPushStrategy;
        stPushStrategy.nIntervalTimes = 2000;
        stPushStrategy.nPushCounts = 1;
        stPushStrategy.bPushSameFrame = AX_TRUE;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPushStrategy;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_PUSH_STRATEGY_T);
        itemIndex++;

        //4 crop_encoder
#if 0
        // body_crop_encoder
        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T stBodyCropEncoderThreshold = {0};
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"body_crop_encoder";
        stBodyCropEncoderThreshold.fScaleLeft = 0;
        stBodyCropEncoderThreshold.fScaleRight = 0;
        stBodyCropEncoderThreshold.fScaleTop = 0;
        stBodyCropEncoderThreshold.fScaleBottom = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stBodyCropEncoderThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T);
        itemIndex++;

        // vehicle_crop_encoder
        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T stVehicleCropEncoderThreshold = {0};
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"vehicle_crop_encoder";
        stVehicleCropEncoderThreshold.fScaleLeft = 0;
        stVehicleCropEncoderThreshold.fScaleRight = 0;
        stVehicleCropEncoderThreshold.fScaleTop = 0;
        stVehicleCropEncoderThreshold.fScaleBottom = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stVehicleCropEncoderThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T);
        itemIndex++;

        // cycle_crop_encoder
        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T stCycleCropEncoderThreshold = {0};
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"cycle_crop_encoder";
        stCycleCropEncoderThreshold.fScaleLeft = 0;
        stCycleCropEncoderThreshold.fScaleRight = 0;
        stCycleCropEncoderThreshold.fScaleTop = 0;
        stCycleCropEncoderThreshold.fScaleBottom = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stCycleCropEncoderThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T);
        itemIndex++;

        // face_crop_encoder
        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T stFaceCropEncoderThreshold = {0};
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"face_crop_encoder";
        stFaceCropEncoderThreshold.fScaleLeft = 0;
        stFaceCropEncoderThreshold.fScaleRight = 0;
        stFaceCropEncoderThreshold.fScaleTop = 0;
        stFaceCropEncoderThreshold.fScaleBottom = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stFaceCropEncoderThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T);
        itemIndex++;

        // plate_crop_encoder
        AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T stPlateCropEncoderThreshold = {0};
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"plate_crop_encoder";
        stPlateCropEncoderThreshold.fScaleLeft = 0;
        stPlateCropEncoderThreshold.fScaleRight = 0;
        stPlateCropEncoderThreshold.fScaleTop = 0;
        stPlateCropEncoderThreshold.fScaleBottom = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPlateCropEncoderThreshold;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_CROP_ENCODER_THRESHOLD_CONFIG_T);
        itemIndex++;
#endif
        // push_panorama
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_panorama";
        AX_SKEL_PUSH_PANORAMA_CONFIG_T stPushPanoramaConfig = {0};
        stPushPanoramaConfig.bEnable = AX_FALSE;
        stPushPanoramaConfig.nQuality = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPushPanoramaConfig;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_PUSH_PANORAMA_CONFIG_T);
        itemIndex++;
#if 0
        // push_quality_body
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_quality_body";
        AX_SKEL_ATTR_FILTER_CONFIG_T stBodyAttrFilter = {0};
        stBodyAttrFilter.stCommonAttrFilterConfig.fQuality = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stBodyAttrFilter;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T);
        itemIndex++;

        // push_quality_vehicle
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_quality_vehicle";
        AX_SKEL_ATTR_FILTER_CONFIG_T stVehicleAttrFilter = {0};
        stVehicleAttrFilter.stCommonAttrFilterConfig.fQuality = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stVehicleAttrFilter;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T);
        itemIndex++;

        // push_quality_cycle
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_quality_cycle";
        AX_SKEL_ATTR_FILTER_CONFIG_T stCycleAttrFilter = {0};
        stCycleAttrFilter.stCommonAttrFilterConfig.fQuality = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stCycleAttrFilter;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T);
        itemIndex++;

        // push_quality_face
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_quality_face";
        AX_SKEL_ATTR_FILTER_CONFIG_T stFaceAttrFilter = {0};
        stFaceAttrFilter.stFaceAttrFilterConfig.nWidth = 0;
        stFaceAttrFilter.stFaceAttrFilterConfig.nHeight = 0;
        stFaceAttrFilter.stFaceAttrFilterConfig.stPoseblur.fPitch = 180;
        stFaceAttrFilter.stFaceAttrFilterConfig.stPoseblur.fYaw = 180;
        stFaceAttrFilter.stFaceAttrFilterConfig.stPoseblur.fRoll = 180;
        stFaceAttrFilter.stFaceAttrFilterConfig.stPoseblur.fBlur = 1.0;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T);
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stFaceAttrFilter;
        itemIndex++;

        // push_quality_plate
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_quality_plate";
        AX_SKEL_ATTR_FILTER_CONFIG_T stPlateAttrFilter = {0};
        stPlateAttrFilter.stCommonAttrFilterConfig.fQuality = 0;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPlateAttrFilter;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_ATTR_FILTER_CONFIG_T);
        itemIndex++;
#endif
//#if 0
        // track_enable
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"track_enable";
        AX_SKEL_COMMON_ENABLE_CONFIG_T stTrackEnable = {0};
        stTrackEnable.bEnable = AX_TRUE;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stTrackEnable;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_ENABLE_CONFIG_T);
        itemIndex++;

        //push_enable
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_enable";
        AX_SKEL_COMMON_ENABLE_CONFIG_T stPushEnable = {0};
        stPushEnable.bEnable = AX_TRUE;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPushEnable;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_ENABLE_CONFIG_T);
        itemIndex++;
//#endif
#if 0
        //target_config
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"target_config";
        AX_SKEL_TARGET_CONFIG_T stTarget = {0};

        stTarget.nSize = 1;
        AX_SKEL_TARGET_ITEM_T stItem = {0};
        stItem.pstrObjectCategory = "body";
        stTarget.pstItems = &stItem;

        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stTarget;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_TARGET_CONFIG_T);
        itemIndex++;

        //push_target_config
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_target_config";
        AX_SKEL_TARGET_CONFIG_T stTarget = {0};
        stTarget.nSize = 1;
        AX_SKEL_TARGET_ITEM_T stItem = {0};
        stItem.pstrObjectCategory = "body";
        stTarget.pstItems = &stItem;

        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stTarget;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_TARGET_CONFIG_T);
        itemIndex++;

        // push_attr_always
        stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"push_attr_always";
        AX_SKEL_COMMON_ENABLE_CONFIG_T stPushAttrAlways = {0};
        stPushAttrAlways.bEnable = AX_TRUE;
        stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stPushAttrAlways;
        stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_COMMON_ENABLE_CONFIG_T);
        itemIndex++;
#endif
        //analyzer_attr_config
        // stConfig.pstItems[itemIndex].pstrType = (AX_CHAR *)"analyzer_attr_config";
        // AX_SKEL_ANALYZER_CONFIG_T stAnalyze = {0};
        // //AX_SKEL_ANALYZER_ATTR_E analyze_attribute = AX_SKEL_ANALYZER_ATTR_NONE;

        // stAnalyze.nSize = 1;
        // AX_SKEL_ANALYZER_ATTR_E analyze_attribute = AX_SKEL_ANALYZER_ATTR_FACE_ATTRIBUTE;
        // stAnalyze.peItems = &analyze_attribute;

        // stConfig.pstItems[itemIndex].pstrValue = (AX_VOID *)&stAnalyze;
        // stConfig.pstItems[itemIndex].nValueSize = sizeof(AX_SKEL_ANALYZER_CONFIG_T);
        // itemIndex++;

        stConfig.nSize = itemIndex;

        nRet = AX_SKEL_SetConfig(pHandle, &stConfig);

        if (0 != nRet) {
            ALOGE("SKEL AX_SKEL_SetConfig, ret = 0x%x", nRet);

            goto EXIT2;
        }
    }

    // get config
    {
        const AX_SKEL_CONFIG_T *pstConfig = NULL;

        nRet = AX_SKEL_GetConfig(pHandle, &pstConfig);

        if (0 != nRet) {
            ALOGE("SKEL AX_SKEL_GetConfig, ret = 0x%x", nRet);

            if (pstConfig) {
                AX_SKEL_Release((AX_VOID *)pstConfig);
            }

            goto EXIT2;
        }

        ParseConfigParam(pstConfig);

        if (pstConfig) {
            AX_SKEL_Release((AX_VOID *)pstConfig);
        }
    }

    ALOGN("Task infomation:");
    ALOGN("\tInput file: %s", InputFile);
    ALOGN("\tInput file resolution: %dx%d", nWidth, nHeight);
    ALOGN("\tRepeat times: %d", nRepeatTimes);
    ALOGN("SKEL Init Elapse:");
    ALOGN("\tAX_SKEL_Init: %lld ms", nInitElasped);
    ALOGN("\tAX_SKEL_Create: %lld ms", nCreateElasped);

    OUTPUT_LOG_SAVE("==============================%s process result:==============================", InputFile);

    if (InputFileFrameCnt > 1) {
        InputFileFrameCnt = InputFileFrameCnt + BLACK_VIDEO_FRAME_COUNT;
    }

    // create frame mgr
    FrameMgrCreate(nFrameSize, nFrameDepth + nCacheListDepth);

    for (AX_U32 nRepeat = 0; nRepeat < nRepeatTimes; nRepeat++) {
        ALOGN("#####SKEL Process times: %d", nRepeat + 1);

        OUTPUT_LOG_SAVE("#####SKEL Process times: %d", nRepeat + 1);

        STAT_OBJECT_NUM_T tObjectTrackNum = {0};
        STAT_OBJECT_NUM_T tObjectPushNum = {0};

        for (AX_U32 nFrameCnt = 0; nFrameCnt < InputFileFrameCnt; nFrameCnt++) {
            AX_BLK blkId = AX_INVALID_BLOCKID;

            if (InputFileFrameCnt != 1) {
                if (FrameSkipCtrl(nSrcFrameRate, nDstFrameRate, nSkelFrameId)) {
                    nSkelFrameId ++;

                    LoadFileToMem(InputFileHandle, 0, NULL, nOneSize);

                    if (nInterval > 0) {
                        usleep(nInterval);
                    }
                    continue;
                }
                else {
                    if (!FrameMgrGet(&YUVDataPhy, &YUVDataVir, nFrameSize, nSkelFrameId, &blkId)) {

                        ALOGE("FrameMgrGet fail");

                        goto EXIT1;
                    }

                    if (nOneSize != nFrameSize) {
                        LoadFileToMem(InputFileHandle, OneYUVDataPhy, OneYUVDataVir, nOneSize);

                        NV12ToStrideNV12(OneYUVDataPhy,
                                        OneYUVDataVir,
                                        nOneSize,
                                        nWidth,
                                        nHeight,
                                        YUVDataPhy,
                                        YUVDataVir,
                                        nFrameSize,
                                        nStride,
                                        nHeight
                                        );
                    }
                    else {
                        LoadFileToMem(InputFileHandle, YUVDataPhy, YUVDataVir, nOneSize);
                    }
                }
            } else {
                FrameMgrGet(&YUVDataPhy, &YUVDataVir, nFrameSize, nSkelFrameId, &blkId);

                if (nRet != 0
                    || YUVDataPhy == 0
                    || YUVDataVir == NULL) {
                    ALOGE("Load input file fail");
                    goto EXIT1;
                }

                if (bJpgFile) {
                    nRet = DecodeJpeg(InputFileHandle, YUVDataPhy, YUVDataVir, nWidth, nHeight);
                    if (0 != nRet) {
                        ALOGE("DecodeJpeg failed!");
                        goto EXIT3;
                    }
                }
                else {
                    if (nOneSize != nFrameSize) {
                        nRet = AX_SYS_MemAlloc(&OneYUVDataPhy, (AX_VOID **)&OneYUVDataVir, nOneSize, 256, (AX_S8 *)"SKEL_TEST");

                        nRet = LoadFileToMem(InputFileHandle, OneYUVDataPhy, OneYUVDataVir, nOneSize);

                        NV12ToStrideNV12(OneYUVDataPhy,
                                        OneYUVDataVir,
                                        nOneSize,
                                        nWidth,
                                        nHeight,
                                        YUVDataPhy,
                                        YUVDataVir,
                                        nFrameSize,
                                        nStride,
                                        nHeight
                                        );
                    }
                    else {
                        nRet = LoadFileToMem(InputFileHandle, YUVDataPhy, YUVDataVir, nFrameSize);
                    }
                }
            }
            stFrame.stFrame.u32Width = nWidth;
            stFrame.stFrame.u32Height = nHeight;
            stFrame.stFrame.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;
            stFrame.stFrame.u32FrameSize = nFrameSize;
            stFrame.stFrame.u32PicStride[0] = nStride;
            stFrame.stFrame.u32PicStride[1] = nStride;
            stFrame.stFrame.u32PicStride[2] = nStride;
            stFrame.stFrame.u64PhyAddr[0] = YUVDataPhy;
            stFrame.stFrame.u64PhyAddr[1] = YUVDataPhy + nStride * nHeight;
            stFrame.stFrame.u64PhyAddr[2] = 0;
            stFrame.stFrame.u64VirAddr[0] = (AX_ULONG)(YUVDataVir);
            stFrame.stFrame.u64VirAddr[1] = (AX_ULONG)(YUVDataVir + nStride * nHeight);
            stFrame.stFrame.u64VirAddr[2] = 0;
            stFrame.stFrame.u32BlkId[0] = blkId;
            stFrame.stFrame.u32BlkId[1] = 0;
            stFrame.stFrame.u32BlkId[2] = 0;
            ALOGN("*****SKEL Frame(%d) Process", nFrameCnt + 1);

            //OUTPUT_LOG_SAVE("*****SKEL Frame(%d) Process Start", nFrameCnt + 1);

            stFrame.nFrameId = nSkelFrameId ++;
            nStartTime = get_tick_count();

            while (1) {
                nRet = AX_SKEL_SendFrame(pHandle, &stFrame, 0);

                if (nRet == 0) {
                    break;
                }

                usleep(1000);
            }

            FrameMgrRelease(blkId);

            nProcessElasped = get_tick_count() - nStartTime;

            if (0 != nRet) {
                ALOGE("SKEL Process fail, ret = 0x%x", nRet);

                goto EXIT3;
            }

            ALOGN("SKEL Process Elapse:");
            ALOGN("\tAX_SKEL_SendFrame: %lld ms", nProcessElasped);

            nStartTime = get_tick_count();

            nRet = AX_SKEL_GetResult(pHandle, &pstResult, -1);

            nResultElasped = get_tick_count() - nStartTime;

            if (0 != nRet) {
                ALOGE("SKEL get result fail, ret = 0x%x", nRet);

                goto EXIT3;
            }

            ALOGN("\tAX_SKEL_GetResult: %lld ms", nResultElasped);

            nResultElaspedTotal += nResultElasped;

            if (nProcessElasped + nResultElasped > nResultElaspedMax) {
                nResultElaspedMax = nProcessElasped + nResultElasped;
            }

            if (nProcessElasped + nResultElasped < nResultElaspedMin) {
                nResultElaspedMin = nProcessElasped + nResultElasped;
            }

            ALOGN("SKEL Process Result:");

            ALOGI("\tFrameId: %lld", pstResult->nFrameId);
            ALOGI("\tnOriginal WxH: %dx%d", pstResult->nOriginalWidth, pstResult->nOriginalHeight);

            ALOGN("\tObject Num: %d", pstResult->nObjectSize);

            AX_U32 nSkelSize = 0;
            AI_Detection_SkelResult_t Skels[SKEL_SAMPLE_OBJECT_SIZE] = {0};
            for (AX_U32 i = 0; i < pstResult->nObjectSize; i++) {
                AX_SKEL_OBJECT_ITEM_T *pstItems = &pstResult->pstObjectItems[i];

                ALOGI("\t\tFrameId: %lld", pstItems->nFrameId);
                ALOGI("\t\tTrackId: %lld, TrackState: %d", pstItems->nTrackId, pstItems->eTrackState);

                ALOGN("\t\tRect[%d] %s: [%f, %f, %f, %f], Confidence: %f", i, pstItems->pstrObjectCategory,
                    pstItems->stRect.fX,
                    pstItems->stRect.fY, pstItems->stRect.fW,
                    pstItems->stRect.fH, pstItems->fConfidence);
                StatTrackMgr(pstItems, &tObjectTrackNum);
                StatPushMgr(pstItems, &tObjectPushNum);

                // get detect box only new or update state
                if ((pstItems->eTrackState == AX_SKEL_TRACK_STATUS_NEW
                    || pstItems->eTrackState == AX_SKEL_TRACK_STATUS_UPDATE)
                    && nSkelSize < SKEL_SAMPLE_OBJECT_SIZE) {
                    Skels[nSkelSize].pstrObjectCategory = pstItems->pstrObjectCategory;
                    Skels[nSkelSize].tBox.fX = pstItems->stRect.fX;
                    Skels[nSkelSize].tBox.fY = pstItems->stRect.fY;
                    Skels[nSkelSize].tBox.fW = pstItems->stRect.fW;
                    Skels[nSkelSize].tBox.fH = pstItems->stRect.fH;

                    OUTPUT_LOG_SAVE("\t\tObject[%d] %s: [%f, %f, %f, %f], Confidence: %f",
                                    i, pstItems->pstrObjectCategory,
                                    pstItems->stRect.fX, pstItems->stRect.fY,
                                    pstItems->stRect.fW, pstItems->stRect.fH,
                                    pstItems->fConfidence);

                    ALOGN("\t\t[%d]Point Set Size: %d", i, pstItems->nPointSetSize);

                    // point
                    Skels[nSkelSize].nPointNum = AX_MIN(DETECT_SKEL_POINT_COUNT, pstItems->nPointSetSize);
                    for (AX_U32 j = 0; j < Skels[i].nPointNum; j++) {
                        ALOGI("\t\t\tPoint[%d] %s: [%f, %f] Confidence: %f", j, pstItems->pstPointSet[j].pstrObjectCategory,
                            pstItems->pstPointSet[j].stPoint.fX,
                            pstItems->pstPointSet[j].stPoint.fY,
                            pstItems->pstPointSet[j].fConfidence);
                            Skels[nSkelSize].tPoint[j].fX = pstItems->pstPointSet[j].stPoint.fX;
                            Skels[nSkelSize].tPoint[j].fY = pstItems->pstPointSet[j].stPoint.fY;

                        OUTPUT_LOG_SAVE("\t\t\tPoint[%d] %s: [%f, %f] Confidence: %f\n",
                                        j,
                                        pstItems->pstPointSet[j].pstrObjectCategory,
                                        pstItems->pstPointSet[j].stPoint.fX,
                                        pstItems->pstPointSet[j].stPoint.fY,
                                        pstItems->pstPointSet[j].fConfidence);
                    }

                    //face attr
                    if(nPPL == AX_SKEL_PPL_FACE){
                        ALOGN("\t\tFace Attribute:");
                        ALOGI("\t\t\tfYaw %f,",pstItems->stFaceAttr.fYaw);
                        ALOGI("\t\t\tfPitch %f,",pstItems->stFaceAttr.fPitch);
                        ALOGI("\t\t\tfRoll %f,",pstItems->stFaceAttr.fRoll);
                        ALOGI("\t\t\tfMask %f,",pstItems->stFaceAttr.fMask);
                        ALOGI("\t\t\tage %d",pstItems->stFaceAttr.nAge);
                        ALOGI("\t\t\tgender %d",pstItems->stFaceAttr.nGender);
                    }
                    nSkelSize ++;
                }

                if (AX_SKEL_TRACK_STATUS_SELECT == pstItems->eTrackState) {
                    ATTRINFO_T AttrInfo;
                    memset(&AttrInfo, 0x00, sizeof(AttrInfo));
                    AttrParser(pstItems, &AttrInfo);

                    if (pstItems->bCropFrame
                        && pstItems->stCropFrame.pFrameData
                        && 0 < pstItems->stCropFrame.nFrameDataSize
                        && SaveResultPath) {
                        // save attribute
                        AX_CHAR arrDat[256] = {0};
                        AX_CHAR strFile[512] = {0};
                        if (AttrInfo.bExist && AttrInfo.eType == ATTR_TYPE_FACE) {
                            sprintf(arrDat, "frame%lld_crop_%s[%lld]_Gender[%s]_Age[%d]_Mask[%s]_Score[%f]_%dx%d",
                                    pstItems->stCropFrame.nFrameId,
                                    pstItems->pstrObjectCategory,
                                    pstItems->nTrackId,
                                    AttrInfo.tFaceInfo.szGender,
                                    AttrInfo.tFaceInfo.nAge,
                                    AttrInfo.tFaceInfo.szMask,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth,
                                    pstItems->stCropFrame.nFrameHeight);

                            OUTPUT_LOG_SAVE("\t\t[FACE ATTR] %s", arrDat);
                        }
                        else if (AttrInfo.bExist && AttrInfo.eType == ATTR_TYPE_PLATE) {
                            sprintf(arrDat, "frame%lld_crop_%s[%lld]_Valid[%d]_Num[%s]_Color[%s]_Score[%f]_%dx%d",
                                    pstItems->stCropFrame.nFrameId,
                                    pstItems->pstrObjectCategory,
                                    pstItems->nTrackId,
                                    AttrInfo.tPlateInfo.bValid,
                                    AttrInfo.tPlateInfo.szNum,
                                    AttrInfo.tPlateInfo.szColor,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth,
                                    pstItems->stCropFrame.nFrameHeight);

                            OUTPUT_LOG_SAVE("\t\t[PLATE ATTR] %s", arrDat);
                        }
                        else if (AttrInfo.bExist && AttrInfo.eType == ATTR_TYPE_VEHICLE) {
                            sprintf(arrDat, "frame%lld_crop_%s[%lld]_[plate_Valid[%d]_Num[%s]_Color[%s]]_Score[%f]_%dx%d",
                                    pstItems->stCropFrame.nFrameId,
                                    pstItems->pstrObjectCategory,
                                    pstItems->nTrackId,
                                    AttrInfo.tPlateInfo.bValid,
                                    AttrInfo.tPlateInfo.szNum,
                                    AttrInfo.tPlateInfo.szColor,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth,
                                    pstItems->stCropFrame.nFrameHeight);

                            OUTPUT_LOG_SAVE("\t\t[VEHICLE ATTR] %s", arrDat);
                        }
                        else {
                            sprintf(arrDat, "frame%lld_crop_%s[%lld]_Score[%f]_%dx%d",
                                    pstItems->stCropFrame.nFrameId,
                                    pstItems->pstrObjectCategory,
                                    pstItems->nTrackId,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth,
                                    pstItems->stCropFrame.nFrameHeight);
                        }

                        sprintf(strFile, "%s/%s/%s.jpg",
                                SaveResultPath,
                                pstItems->pstrObjectCategory,
                                arrDat);

                        FILE *fp_w = fopen(strFile, "wb");

                        if (fp_w) {
                            ALOGI("Write crop jpg to file: %s", strFile);
                            fwrite((AX_U8 *)pstItems->stCropFrame.pFrameData, 1, pstItems->stCropFrame.nFrameDataSize, fp_w);
                            fclose(fp_w);
                        }

                    }

                    // panora frame
                    if (pstItems->bPanoraFrame
                        && pstItems->stPanoraFrame.pFrameData
                        && 0 < pstItems->stPanoraFrame.nFrameDataSize
                        && SaveResultPath) {
                        AX_CHAR arrDat[256] = {0};
                        AX_CHAR strFile[512] = {0};
                        if (AttrInfo.bExist && AttrInfo.eType == ATTR_TYPE_FACE) {
                            sprintf(arrDat, "frame%lld_panora_%s[%lld]_Gender[%s]_Age[%d]_Mask[%s]_Score[%f]_%dx%d.jpg",
                                    pstItems->stCropFrame.nFrameId, pstItems->pstrObjectCategory, pstItems->nTrackId,
                                    AttrInfo.tFaceInfo.szGender, AttrInfo.tFaceInfo.nAge, AttrInfo.tFaceInfo.szMask,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth, pstItems->stCropFrame.nFrameHeight);
                        }
                        else if (AttrInfo.bExist && AttrInfo.eType == ATTR_TYPE_PLATE) {
                            sprintf(arrDat, "frame%lld_panora_%s[%lld]_Valid[%d]_Num[%s]_Color[%s]_Score[%f]_%dx%d.jpg",
                                    pstItems->stCropFrame.nFrameId, pstItems->pstrObjectCategory, pstItems->nTrackId,
                                    AttrInfo.tPlateInfo.bValid, AttrInfo.tPlateInfo.szNum, AttrInfo.tPlateInfo.szColor,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth, pstItems->stCropFrame.nFrameHeight);
                        }
                        else if (AttrInfo.bExist && AttrInfo.eType == ATTR_TYPE_VEHICLE) {
                            sprintf(arrDat, "frame%lld_panora_%s[%lld]_[plate_Valid[%d]_Num[%s]_Color[%s]]_Score[%f]_%dx%d.jpg",
                                    pstItems->stCropFrame.nFrameId, pstItems->pstrObjectCategory, pstItems->nTrackId,
                                    AttrInfo.tPlateInfo.bValid, AttrInfo.tPlateInfo.szNum, AttrInfo.tPlateInfo.szColor,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth, pstItems->stCropFrame.nFrameHeight);
                        }
                        else {
                            sprintf(arrDat, "frame%lld_panora_%s[%lld]_Score[%f]_%dx%d.jpg",
                                    pstItems->stCropFrame.nFrameId, pstItems->pstrObjectCategory, pstItems->nTrackId,
                                    pstItems->fConfidence,
                                    pstItems->stCropFrame.nFrameWidth, pstItems->stCropFrame.nFrameHeight);
                        }

                        sprintf(strFile, "%s/%s/%s.jpg",
                                SaveResultPath,
                                pstItems->pstrObjectCategory,
                                arrDat);

                        FILE *fp_w = fopen(strFile, "wb");

                        if (fp_w) {
                            ALOGI("Write panora jpg to file: %s", strFile);
                            fwrite((AX_U8 *)pstItems->stPanoraFrame.pFrameData, 1, pstItems->stPanoraFrame.nFrameDataSize, fp_w);
                            fclose(fp_w);
                        }
                    }
                }

                // feature
                ALOGI("\t\tFeature Size: %d", pstItems->nFeatureSize);
                if (pstItems->pstFeatureItem
                    && 0 < pstItems->nFeatureSize
                    && SaveResultPath) {
                    AX_CHAR strFile[256] = {0};
                    sprintf(strFile, "%s/%s/frame%lld_feature_%s_%d.db",
                        SaveResultPath, pstItems->pstrObjectCategory,
                        pstItems->stCropFrame.nFrameId,
                        pstItems->pstrObjectCategory, i);
                    FILE *fp_w = fopen(strFile, "wb");

                    if (fp_w) {
                        ALOGI("\t\t\tWrite feature to file: %s, nValueSize: %d",
                                strFile,
                                pstItems->pstFeatureItem[0].nValueSize);
                        fwrite((AX_U8 *)pstItems->pstFeatureItem[0].pstrValue, 1, pstItems->pstFeatureItem[0].nValueSize, fp_w);
                        fclose(fp_w);
                    }
                }

                // meta
                ALOGI("\t\tMeta Size: %d", pstItems->nMetaInfoSize);
                if (pstItems->pstMetaInfo
                    && 0 < pstItems->nMetaInfoSize) {
                    for (AX_U32 i = 0; i < pstItems->nMetaInfoSize; i++) {
                        ALOGI("\t\tMetaInfo[%d] %s: \n%s", i, pstItems->pstMetaInfo[i].pstrType, pstItems->pstMetaInfo[i].pstrValue);
                    }
                }

                // binds
                ALOGI("\t\tBind Size: %d", pstItems->nObjectBindSize);
                if (pstItems->pstObjectBind
                    && 0 < pstItems->nObjectBindSize) {
                    for (AX_U32 i = 0; i < pstItems->nObjectBindSize; i++) {
                        ALOGI("\t\t[%s] [TrackId] %lld bind to ObjectBind[%d]: %s [TrackId] %lld",
                            pstItems->pstrObjectCategory, pstItems->nTrackId, i,
                            pstItems->pstObjectBind[i].pstrObjectCategoryBind, pstItems->pstObjectBind[i].nTrackId);
                    }
                }
            }

            ALOGI("\tnCacheListSize: %d", pstResult->nCacheListSize);
            if (pstResult->pstCacheList) {
                for (AX_U32 i = 0; i < pstResult->nCacheListSize; i++) {
                    ALOGI("\t\tCacheList[%d] FrameId: %lld", i, pstResult->pstCacheList[i].nFrameId);
                }
            }

            if (ReWritePath && pstResult->nObjectSize > 0) {
                memcpy(YUVDataWrVir, YUVDataVir, nFrameSize);

                YUV_IMAGE_T YUVImage = {0};
                YUVImage.pImage = YUVDataWrVir;
                YUVImage.nWidth = nWidth;
                YUVImage.nHeight = nHeight;
                YUVImage.stride = nStride;
                YUVImage.nSize = nFrameSize;
                YUVImage.eType = AX_FORMAT_YUV420_SEMIPLANAR;

                for (size_t i = 0; i < nSkelSize; i++) {
                    // draw rect
                    AX_CHAR ObjectCategory = Skels[i].pstrObjectCategory[0];
                    AX_S16 x0 = Skels[i].tBox.fX;
                    AX_S16 y0 = Skels[i].tBox.fY;
                    AX_U16 w = Skels[i].tBox.fW;
                    AX_U16 h = Skels[i].tBox.fH;
                    YUV_COLOR RectColor = YUV_WHITE;

                    switch(ObjectCategory){
                        case 'b': //body
                        RectColor = YUV_WHITE;
                        break;
                        case 'v': //vehicle
                        RectColor = YUV_PURPLE;
                        break;
                        case 'c': //cycle
                        RectColor = YUV_GREEN;
                        break;
                        case 'f': //face
                        RectColor = YUV_YELLOW;
                        break;
                        case 'p': //plate
                        RectColor = YUV_RED;
                        break;
                        default :
                        RectColor = YUV_WHITE;
                        break;
                    }

                    DrawRect(&YUVImage, x0, y0, w, h, RectColor);

                    if (Skels[i].nPointNum > 0) {
                        // draw point
                        for (size_t j = 0; j < Skels[i].nPointNum; j++) {
                            x0 = Skels[i].tPoint[j].fX;
                            y0 = Skels[i].tPoint[j].fY;

                            DrawPoint(&YUVImage, x0, y0, 4, x0 * (4 - 1), y0 * (4 - 1), YUV_DARK_GREEN);
                        }
                    }
                }

                AX_CHAR arrWrFile[512] = {0};
                sprintf(arrWrFile, "%s_result_%d_%d.jpg", InputFile, nFrameCnt + 1, nRepeat + 1);

                AX_CHAR *fileName = NULL;
                AX_U32 fileNameLen = strlen(arrWrFile);
                AX_CHAR fullFileName[256] = {0};
                AX_S32 i = fileNameLen - 1;

                if (fileNameLen > 0) {
                    for (; i >= 0; i --) {
                        if (arrWrFile[i] == '/') {
                            break;
                        }
                    }
                }

                if (i < 0) {
                    i = 0;
                }
                else if (i < fileNameLen - 1) {
                    i ++;
                }

                fileName = (AX_CHAR *)&arrWrFile[i];

                sprintf(fullFileName, "%s/%s", ReWritePath, fileName);

                EncodeOneFrameToJpeg((AX_CHAR *)fullFileName, nStride, nWidth, nHeight, YUVDataWrPhy, YUVDataWrVir, nFrameSize);
            }

            if (pstResult) {
                AX_SKEL_Release((AX_VOID *)pstResult);
            }

            OUTPUT_LOG_SAVE("*****SKEL Frame(%d) Process End (Elasped: %lld ms)", nFrameCnt + 1, nProcessElasped);

            if (nInterval > 0) {
                usleep(nInterval);
            }
        }

        ALOGN("SKEL Process Objects Statistics: Body[%d], Vehicle[%d], Cycle[%d], Face[%d], Plate[%d]",
                tObjectTrackNum.nBodyNum,
                tObjectTrackNum.nVehicleNum,
                tObjectTrackNum.nCycleNum,
                tObjectTrackNum.nFaceNum,
                tObjectTrackNum.nPlateNum);

        OUTPUT_LOG_SAVE("\nSKEL Process Objects Statistics: Body[%d], Vehicle[%d], Cycle[%d], Face[%d], Plate[%d]",
                        tObjectTrackNum.nBodyNum,
                        tObjectTrackNum.nVehicleNum,
                        tObjectTrackNum.nCycleNum,
                        tObjectTrackNum.nFaceNum,
                        tObjectTrackNum.nPlateNum);

        ALOGN("SKEL Process Push Statistics: Body[%d], Vehicle[%d], Cycle[%d], Face[%d], Plate[%d]",
                tObjectPushNum.nBodyNum,
                tObjectPushNum.nVehicleNum,
                tObjectPushNum.nCycleNum,
                tObjectPushNum.nFaceNum,
                tObjectPushNum.nPlateNum);

        OUTPUT_LOG_SAVE("SKEL Process Push Statistics: Body[%d], Vehicle[%d], Cycle[%d], Face[%d], Plate[%d]\n",
                        tObjectPushNum.nBodyNum,
                        tObjectPushNum.nVehicleNum,
                        tObjectPushNum.nCycleNum,
                        tObjectPushNum.nFaceNum,
                        tObjectPushNum.nPlateNum);

        if (InputFileHandle) {
            ReLoadFile(InputFileHandle);
        }
    }

    // if(AX_SUCCESS != nRet){
    //     ALOGE("AX_POOL_DestroyPool failed! Error Code:0x%X\n", nRet);
    // }

    ALOGN("SKEL Process Elapsed Info: Repeats: %d, (min: %lld ms, avr: %lld ms, max: %lld ms)",
            nRepeatTimes,
            nResultElaspedMin,
            (nSkelFrameId > 1) ? (nResultElaspedTotal / (nSkelFrameId - 1)) : 0,
            nResultElaspedMax);

    OUTPUT_LOG_SAVE("SKEL Process Elapsed Info: Repeats: %d, (min: %lld ms, avr: %lld ms, max: %lld ms)",
                    nRepeatTimes,
                    nResultElaspedMin,
                    (nSkelFrameId > 1) ? (nResultElaspedTotal / (nSkelFrameId - 1)) : 0,
                    nResultElaspedMax);

EXIT3:
    if (pHandle) {
        AX_SKEL_Destroy(pHandle);
    }

EXIT2:
    AX_SKEL_DeInit();

EXIT1:
    AX_ENGINE_Deinit();

EXIT0:
    // if (YUVDataVir) {
    //     if (YUVDataPhy == 0) {
    //         free(YUVDataVir);
    //     } else {
    //         AX_SYS_MemFree(YUVDataPhy, YUVDataVir);
    //     }
    // }
    if (YUVDataWrVir) {
        if (YUVDataWrPhy == 0) {
            free(YUVDataWrVir);
        } else {
            AX_SYS_MemFree(YUVDataWrPhy, YUVDataWrVir);
        }
    }

    if (OneYUVDataVir) {
        if (OneYUVDataPhy == 0) {
            free(OneYUVDataVir);
        } else {
            AX_SYS_MemFree(OneYUVDataPhy, OneYUVDataVir);
        }
    }

    FrameMgrDestroy();

    if (fpResultFile) {
        fclose(fpResultFile);
    }

    AX_VDEC_Deinit();

    AX_VENC_Deinit();

    AX_IVPS_Deinit();

    AX_SYS_Deinit();

    return (0 != nRet) ? -1 : 0;
}
