/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>
#include <pthread.h>
#define CONFIG_CAPTURE_BUF_CNT 10
#define CONFIG_DEVNAME_LEN     32

#if __cplusplus
extern "C" {
#endif
typedef void (*vcamera_frame_get)(void* pdata, uint32_t width, uint32_t height, uint32_t length, void* ctx);
typedef enum { CAMERA_STATE_CLOSE, CAMERA_SATTE_OPEN, CAMERA_SATTE_CAP, CAMERA_STATE_ERR } CAMERA_STATE_t;

typedef struct {
    void* pstart;
    int length;
} CAMERA_Buffer_t;

typedef struct camera_t {
    char dev_name_[CONFIG_DEVNAME_LEN];
    int dev_fd_;
    int buffer_cnt_;
    int capture_fps_;
    int is_alloc_;
    uint32_t width_;
    uint32_t height_;
    CAMERA_Buffer_t* pcamera_buffer_;
    vcamera_frame_get pcallback_;
    int state_; /* 0 Not open, 1 Turn on the camera, 2 Start capturing,
                            3 Error */
    pthread_t capture_thread_id_;
    void* ctx_;
    void* custom_config_;
    /**
     * Set capture frame callback
     * Return value: 0 for success, -1 for failure
     */
    int (*camera_capture_callback_set)(struct camera_t*, vcamera_frame_get);
    /**
     * Start capturing
     * Return value: 0 for success, -1 for failure
     */
    int (*camera_capture_start)(struct camera_t*);
    /**
     * Stop capturing
     * Return value: 0 for success, -1 for failure
     */
    int (*camera_capture_stop)(struct camera_t*);
    void (*camera_set_ctx)(struct camera_t*, void*);
} camera_t;

/**
 * Open the camera
 * @pdev_name Device node
 * Return value: NULL for failure
 */
camera_t* camera_open(const char* pdev_name, int width, int height, int fps, void* config);

/**
 * Open the camera from config
 * @pDevName Device node
 * Return value: 0 for success, -1 for failure
 */
int camera_open_from(camera_t* camera);

/**
 * Close the camera
 * Return value: 0 for success, -1 for failure
 */
int camera_close(camera_t* camera);

#if __cplusplus
}
#endif
#endif