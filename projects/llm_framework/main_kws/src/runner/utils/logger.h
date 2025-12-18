/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/

#ifndef SKEL_LOGGER_H
#define SKEL_LOGGER_H

#include "ax_global_type.h"
#include "ax_sys_log.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// #define SKEL_LOG_TAG "SKEL"
//
// #define ALOGE(fmt, ...) AX_LOG_ERR_EX(SKEL_LOG_TAG, AX_ID_SKEL, fmt, ##__VA_ARGS__)
// #define ALOGW(fmt, ...) AX_LOG_WARN_EX(SKEL_LOG_TAG, AX_ID_SKEL, fmt, ##__VA_ARGS__)
// #define ALOGI(fmt, ...) AX_LOG_INFO_EX(SKEL_LOG_TAG, AX_ID_SKEL, fmt, ##__VA_ARGS__)
// #define ALOGD(fmt, ...) AX_LOG_DBG_EX(SKEL_LOG_TAG, AX_ID_SKEL, fmt, ##__VA_ARGS__)
// #define ALOGN(fmt, ...) AX_LOG_NOTICE_EX(SKEL_LOG_TAG, AX_ID_SKEL, fmt, ##__VA_ARGS__)

typedef enum {
    SKEL_LOG_MIN       = -1,
    SKEL_LOG_EMERGENCY = 0,
    SKEL_LOG_ALERT     = 1,
    SKEL_LOG_CRITICAL  = 2,
    SKEL_LOG_ERROR     = 3,
    SKEL_LOG_WARN      = 4,
    SKEL_LOG_NOTICE    = 5,
    SKEL_LOG_INFO      = 6,
    SKEL_LOG_DEBUG     = 7,
    SKEL_LOG_MAX
} SKEL_LOG_LEVEL_E;

static SKEL_LOG_LEVEL_E log_level = SKEL_LOG_DEBUG;

#if 1
#define MACRO_BLACK  "\033[1;30;30m"
#define MACRO_RED    "\033[1;30;31m"
#define MACRO_GREEN  "\033[1;30;32m"
#define MACRO_YELLOW "\033[1;30;33m"
#define MACRO_BLUE   "\033[1;30;34m"
#define MACRO_PURPLE "\033[1;30;35m"
#define MACRO_WHITE  "\033[1;30;37m"
#define MACRO_END    "\033[0m"
#else
#define MACRO_BLACK
#define MACRO_RED
#define MACRO_GREEN
#define MACRO_YELLOW
#define MACRO_BLUE
#define MACRO_PURPLE
#define MACRO_WHITE
#define MACRO_END
#endif

#define ALOGE(fmt, ...) printf(MACRO_RED "[E][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGW(fmt, ...)             \
    if (log_level >= SKEL_LOG_WARN) \
    printf(MACRO_YELLOW "[W][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGI(fmt, ...)             \
    if (log_level >= SKEL_LOG_INFO) \
    printf(MACRO_GREEN "[I][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGD(fmt, ...)              \
    if (log_level >= SKEL_LOG_DEBUG) \
    printf(MACRO_WHITE "[D][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define ALOGN(fmt, ...)               \
    if (log_level >= SKEL_LOG_NOTICE) \
    printf(MACRO_PURPLE "[N][%32s][%4d]: " fmt MACRO_END "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif  // SKEL_LOGGER_H
