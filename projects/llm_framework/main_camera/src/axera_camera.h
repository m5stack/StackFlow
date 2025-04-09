/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef AXERA_CAMERA_H
#define AXERA_CAMERA_H
#include "common_venc.h"
#if __cplusplus
extern "C" {
#endif
/**
 * Open the axera_camera
 * @pdev_name Device node
 * Return value: NULL for failure
 */
camera_t* axera_camera_open(const char* pdev_name, int width, int height, int fps);

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
void init_jpeg();

#if __cplusplus
}
#endif
#endif