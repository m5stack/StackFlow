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

static int camera_capture_callback_set(struct camera_t* camera, vcamera_frame_get pcallback)
{
    if (camera->state_ == CAMERA_SATTE_CAP) {
        SLOGW("Set capture callback failed");
        return -1;
    }
    camera->pcallback_ = pcallback;
    return 0;
}

static void* camera_capture_thread(void* param)
{
    int Ret          = -1;
    camera_t* camera = (camera_t*)param;
    struct v4l2_buffer EnQueueBuf;
    struct v4l2_buffer DeQueueBuf;

    SLOGI("Start capture");

    while (camera->state_ == CAMERA_SATTE_CAP) {
        memset(&DeQueueBuf, 0, sizeof(struct v4l2_buffer));
        DeQueueBuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        DeQueueBuf.memory = V4L2_MEMORY_MMAP;

        memset(&EnQueueBuf, 0, sizeof(struct v4l2_buffer));
        EnQueueBuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        EnQueueBuf.memory = V4L2_MEMORY_MMAP;

        // Retrieve data from the queue
        Ret = ioctl(camera->dev_fd_, VIDIOC_DQBUF, &DeQueueBuf);
        if (Ret) {
            perror("Dequeue fail");
            break;
        }
        camera->pcallback_(camera->pcamera_buffer_[DeQueueBuf.index].pstart, camera->width_, camera->height_,
                           camera->pcamera_buffer_[DeQueueBuf.index].length, camera->ctx_);

        // Re-queue the data
        EnQueueBuf.index = DeQueueBuf.index;

        if (ioctl(camera->dev_fd_, VIDIOC_QBUF, &EnQueueBuf)) {
            perror("Enqueue fail");
            break;
        }
    }

    SLOGI("Stop capture");

    return NULL;
}

static int camera_capture_start(struct camera_t* camera)
{
    SLOGI("Start capture thread");
    if (!camera->pcallback_) {
        SLOGW("Capture callback not set, start faild");
        return -1;
    }
    if (camera->state_ != CAMERA_SATTE_OPEN) {
        SLOGW("Camera not open, start faild");
        return -1;
    }
    camera->state_ = CAMERA_SATTE_CAP;

    pthread_create(&camera->capture_thread_id_, NULL, camera_capture_thread, camera);

    return 0;
}

static int camera_capture_stop(struct camera_t* camera)
{
    SLOGI("Stop capture thread");
    camera->state_ = CAMERA_SATTE_OPEN;
    pthread_join(camera->capture_thread_id_, NULL);
    SLOGI("Capture thread stop");
    return 0;
}

static void camera_set_ctx(struct camera_t* camera, void* ctx)
{
    camera->ctx_ = ctx;
}

int camera_open_from(camera_t* camera)
{
    int Ret = -1;
    if (camera == NULL) return -1;
    /* Check whether the camera is already open or in an error state */
    SLOGI("Open camera %s...", camera->dev_name_);
    if (camera->state_ != CAMERA_STATE_CLOSE) {
        SLOGE("Error: camera was open or meet error, now state is: %d", camera->state_);
        goto ErrorHandle;
    }

    /* Open the camera device */
    if ((camera->dev_fd_ = open(camera->dev_name_, O_RDWR, 0)) == -1) {
        SLOGE("Open camera %s faild.", camera->dev_name_);
        goto ErrorHandle;
    }

    SLOGI("Open device successful.");

    /* Set the capture source */
    struct v4l2_input Input;
    Input.index = 0;

    if (ioctl(camera->dev_fd_, VIDIOC_S_INPUT, &Input)) {
        perror("VIDIOC_S_INPUT");
    }

    /* Retrieve camera information */
    struct v4l2_capability Cap;

    Ret = ioctl(camera->dev_fd_, VIDIOC_QUERYCAP, &Cap);

    if (Ret) {
        perror("Get device info error");
        goto ErrorHandle;
    }

    SLOGI("Get device info successful.");

    SLOGI("Driver name: %s", Cap.driver);
    SLOGI("Card name: %s", Cap.card);
    SLOGI("Bus info: %s", Cap.bus_info);

    struct v4l2_format TvFmt;
    struct v4l2_fmtdesc FmtDesc;
    struct v4l2_streamparm StreamParam;

    memset(&FmtDesc, 0, sizeof(FmtDesc));
    FmtDesc.index = 0;
    FmtDesc.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    /* Enumerate supported frame formats */
    SLOGI("Support format:");
    while (ioctl(camera->dev_fd_, VIDIOC_ENUM_FMT, &FmtDesc) == 0) {
        FmtDesc.index++;
        SLOGI("Pixel format = %c%c%c%c", (FmtDesc.pixelformat & 0xFF), ((FmtDesc.pixelformat >> 8) & 0xFF),
              ((FmtDesc.pixelformat >> 16) & 0xFF), (FmtDesc.pixelformat >> 24));
        SLOGI("Description = %s ", FmtDesc.description);
    }

    /* Determine if the device is a device that can capture video */
    if (!(Cap.capabilities & V4L2_BUF_TYPE_VIDEO_CAPTURE)) {
        SLOGE("The Current device is not a video capture device");
        goto ErrorHandle;
    }

    /* Determine if video streaming is supported */
    if (!(Cap.capabilities & V4L2_CAP_STREAMING)) {
        SLOGE("The Current device does not support streaming i/o");
        goto ErrorHandle;
    }

    /* Get video stream information */
    memset(&StreamParam, 0, sizeof(struct v4l2_streamparm));
    StreamParam.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    Ret              = ioctl(camera->dev_fd_, VIDIOC_G_PARM, &StreamParam);
    if (Ret) {
        perror("Get stream info failed");
        goto ErrorHandle;
    }

    SLOGI("Before setting stream params:");
    SLOGI("Capability: %u", StreamParam.parm.capture.capability);
    SLOGI("Capturemode: %u", StreamParam.parm.capture.capturemode);
    SLOGI("Extendedmode: %u", StreamParam.parm.capture.extendedmode);
    SLOGI("Timeperframe denominator: %u", StreamParam.parm.capture.timeperframe.denominator);
    SLOGI("Timeperframe numerator: %u", StreamParam.parm.capture.timeperframe.numerator);

    /* Set frame rate denominator and numerator */
    StreamParam.parm.capture.timeperframe.denominator = camera->capture_fps_;
    StreamParam.parm.capture.timeperframe.numerator   = 1;

    if (ioctl(camera->dev_fd_, VIDIOC_S_PARM, &StreamParam) == -1) {
        perror("Unable to set fps");
        goto ErrorHandle;
    }

    /* Get video stream information */
    Ret = ioctl(camera->dev_fd_, VIDIOC_G_PARM, &StreamParam);
    if (Ret) {
        perror("Get stream info failed");
        goto ErrorHandle;
    }

    SLOGI("After setting stream params:");
    SLOGI("Capability: %u", StreamParam.parm.capture.capability);
    SLOGI("Capturemode: %u", StreamParam.parm.capture.capturemode);
    SLOGI("Extendedmode: %u", StreamParam.parm.capture.extendedmode);
    SLOGI("Timeperframe denominator: %u", StreamParam.parm.capture.timeperframe.denominator);
    SLOGI("Timeperframe numerator: %u", StreamParam.parm.capture.timeperframe.numerator);

    /* Set automatic exposure */
    struct v4l2_control Ctrl;
    Ctrl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
    Ret     = ioctl(camera->dev_fd_, VIDIOC_G_CTRL, &Ctrl);

    if (Ret) {
        perror("Set exposure failed");
        // goto ErrorHandle;
    }

    /* Set capture format */
    TvFmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    TvFmt.fmt.pix.width       = camera->width_;
    TvFmt.fmt.pix.height      = camera->height_;
    TvFmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    TvFmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;

    if (ioctl(camera->dev_fd_, VIDIOC_S_FMT, &TvFmt) < 0) {
        perror("Unable set VIDIOC_S_FMT");
        goto ErrorHandle;
    }

    if (ioctl(camera->dev_fd_, VIDIOC_G_FMT, &TvFmt) < 0) {
        perror("Get VIDIOC_S_FMT error");
        goto ErrorHandle;
    }

    SLOGI("After setting frame params:");
    SLOGI("Width: %d, height: %d", TvFmt.fmt.pix.width, TvFmt.fmt.pix.height);
    SLOGI("Pixel format = %c%c%c%c", (TvFmt.fmt.pix.pixelformat & 0xFF), ((TvFmt.fmt.pix.pixelformat >> 8) & 0xFF),
          ((TvFmt.fmt.pix.pixelformat >> 16) & 0xFF), (TvFmt.fmt.pix.pixelformat >> 24));

    camera->width_  = TvFmt.fmt.pix.width;
    camera->height_ = TvFmt.fmt.pix.height;

    /* Request V4L2 driver to allocate memory */
    struct v4l2_requestbuffers ReqBuffer;
    memset(&ReqBuffer, 0, sizeof(ReqBuffer));
    ReqBuffer.count  = camera->buffer_cnt_;
    ReqBuffer.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ReqBuffer.memory = V4L2_MEMORY_MMAP;

    if (-1 == ioctl(camera->dev_fd_, VIDIOC_REQBUFS, &ReqBuffer)) {
        perror("Fail to ioctl 'VIDIOC_REQBUFS'");
        goto ErrorHandle;
    }

    camera->buffer_cnt_ = ReqBuffer.count;
    SLOGI("Buffer count: %d", camera->buffer_cnt_);

    camera->pcamera_buffer_ = (CAMERA_Buffer_t*)calloc(camera->buffer_cnt_, sizeof(CAMERA_Buffer_t));
    if (camera->pcamera_buffer_ == NULL) {
        SLOGE("Malloc buffer failed");
        goto ErrorHandle;
    }

    /* Map kernel buffer to user space */
    struct v4l2_buffer Buf;
    for (int i = 0; i < camera->buffer_cnt_; i++) {
        memset(&Buf, 0, sizeof(Buf));
        Buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        Buf.memory = V4L2_MEMORY_MMAP;
        Buf.index  = i;

        if (-1 == ioctl(camera->dev_fd_, VIDIOC_QUERYBUF, &Buf)) {
            perror("Fail to ioctl : VIDIOC_QUERYBUF");
            goto ErrorHandle;
        }

        camera->pcamera_buffer_[i].length = Buf.length;
        camera->pcamera_buffer_[i].pstart =
            mmap(NULL, Buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, camera->dev_fd_, Buf.m.offset);

        if (MAP_FAILED == camera->pcamera_buffer_[i].pstart) {
            perror("Fail to mmap");
            goto ErrorHandle;
        }

        if (ioctl(camera->dev_fd_, VIDIOC_QBUF, &Buf)) {
            perror("Unable to queue buffer");
            goto ErrorHandle;
        }
    }

    /* Start capturing */
    enum v4l2_buf_type BufType;
    BufType = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->dev_fd_, VIDIOC_STREAMON, &BufType) < 0) {
        perror("VIDIOC_STREAMON error");
        goto ErrorHandle;
    }
    camera->state_                      = CAMERA_SATTE_OPEN;
    camera->camera_capture_callback_set = camera_capture_callback_set;
    camera->camera_capture_start        = camera_capture_start;
    camera->camera_capture_stop         = camera_capture_stop;
    camera->camera_set_ctx              = camera_set_ctx;
    camera->is_alloc_                   = 0;
    return 0;

ErrorHandle:
    SLOGE("Camera open meet error, now handle it");
    close(camera->dev_fd_);
    free(camera->pcamera_buffer_);
    return -1;
}

camera_t* camera_open(const char* pdev_name, int width, int height, int fps, void* config)
{
    int Ret          = -1;
    camera_t* camera = (camera_t*)malloc(sizeof(camera_t));
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

    camera->width_       = width;
    camera->height_      = height;
    camera->capture_fps_ = fps;

    Ret = camera_open_from(camera);
    if (Ret) {
        goto ErrorHandle;
    }
    camera->is_alloc_ = 1;
    return camera;

ErrorHandle:
    SLOGE("Camera open meet error, now handle it");
    free(camera);
    return NULL;
}

int camera_close(camera_t* camera)
{
    if (camera == NULL) return -1;
    if (camera->state_ == CAMERA_SATTE_OPEN || camera->state_ == CAMERA_SATTE_CAP) {
        SLOGI("Close camera %s", camera->dev_name_);
    } else {
        SLOGW("Skip close camera progress");
        return 0;
    }

    for (int i = 0; i < CONFIG_CAPTURE_BUF_CNT; i++) {
        munmap(camera->pcamera_buffer_[i].pstart, camera->pcamera_buffer_[i].length);
        camera->pcamera_buffer_[i].length = 0;
    }

    free(camera->pcamera_buffer_);
    camera->pcamera_buffer_ = NULL;

    int Off = 1;

    if (ioctl(camera->dev_fd_, VIDIOC_STREAMOFF, &Off)) {
        perror("Stop camera fail");
        return -1;
    }

    close(camera->dev_fd_);
    if (camera->is_alloc_) free(camera);

    SLOGI("camera closed");

    return 0;
}