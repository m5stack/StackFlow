/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "StackFlow.h"
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include "../../../../SDK/components/utilities/include/sample_log.h"
#include "camera.h"
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
#include "axera_camera.h"
#endif
#include <glob.h>
#include <opencv2/opencv.hpp>
#include "hv/TcpServer.h"
#include <sys/time.h>
#include <time.h>

#include <regex>
// #include <jpeglib.h>

#ifdef ENABLE_BACKWARD
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
#include "backward.h"
#endif

#define MAX_TASK_NUM 1
using namespace StackFlows;
int main_exit_flage = 0;

const char *http_response =
    "HTTP/1.0 200 OK\n"
    "Server: BaseHTTP/0.6 Python/3.10.12\n"
    "Date: %s\n"
    "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\n"
    "Connection: close\n"
    "Content-Type: multipart/x-mixed-replace;boundary=--boundarydonotcross\n"
    "Expires: Mon, 1 Jan 2130 00:00:00 GMT\n"
    "Pragma: no-cache\n"
    "Access-Control-Allow-Origin: *\n";
const char *http_jpeg_response =
    "\n"
    "--boundarydonotcross\n"
    "X-Timestamp: %lf\n"
    "Content-Length: %d\n"
    "Content-Type: image/jpeg\n"
    "\n";

char http_response_buff[1024];
char http_response_buff1[1024];

static void __sigint(int iSigNo)
{
    main_exit_flage = 1;
}

typedef std::function<void(const void *, int)> task_callback_t;

typedef camera_t *(*hal_camera_open_fun)(const char *pdev_name, int width, int height, int fps, void *config);
typedef int (*hal_camera_close_fun)(camera_t *camera);
typedef bool (*hal_parse_config_fun)(const nlohmann::json &config_body, const nlohmann::json &file_body,
                                     void **custom_config);

#define CONFIG_OBJECT stVencChnAttr

#define CONFIG_AUTO_SET(obj, key)              \
    if (config_body.contains(#key))            \
        CONFIG_OBJECT.key = config_body[#key]; \
    else if (obj.contains(#key))               \
        CONFIG_OBJECT.key = obj[#key];

#define CONFIG_AUTO_SET_DEFAULT(obj, key, default_value) \
    if (config_body.contains(#key))                      \
        CONFIG_OBJECT.key = config_body[#key];           \
    else if (obj.contains(#key))                         \
        CONFIG_OBJECT.key = obj[#key];                   \
    else                                                 \
        CONFIG_OBJECT.key = default_value;

class llm_task {
private:
    camera_t *cam;
    void *cam_config;
    hal_camera_open_fun hal_camera_open;
    hal_camera_close_fun hal_camera_close;
    hal_parse_config_fun hal_parse_config;

public:
    std::string response_format_;
    task_callback_t out_callback_;
    bool enoutput_;
    bool enstream_;
    bool enjpegout_;
    std::string rtsp_config_;
    bool enable_webstream_;
    std::atomic_int cap_status_;
    std::unique_ptr<std::thread> camera_cap_thread_;
    std::atomic_bool camera_clear_flage_;

    std::string devname_;
    int frame_width_;
    int frame_height_;
    cv::Mat yuv_dist_;

    std::unique_ptr<hv::TcpServer> hv_tcpserver_;

    static void on_cap_fream(void *pData, uint32_t width, uint32_t height, uint32_t Length, void *ctx)
    {
        llm_task *self = static_cast<llm_task *>(ctx);
        int src_offsetX;
        int src_offsetY;
        int src_W;
        int src_H;
        int dst_offsetX;
        int dst_offsetY;
        int dst_W;
        int dst_H;
        if ((self->frame_height_ == height) && (self->frame_width_ == width)) {
            if (self->out_callback_) self->out_callback_(pData, Length);
        } else {
            if ((self->frame_height_ >= height) && (self->frame_width_ >= width)) {
                src_offsetX = 0;
                src_offsetY = 0;
                src_W       = width;
                src_H       = height;
                dst_offsetX = (self->frame_width_ == src_W) ? 0 : (self->frame_width_ - src_W) / 2;
                dst_offsetY = (self->frame_height_ == src_H) ? 0 : (self->frame_height_ - src_H) / 2;
                dst_W       = width;
                dst_H       = height;
            } else if ((self->frame_height_ <= height) && (self->frame_width_ <= width)) {
                src_offsetX = (self->frame_width_ == width) ? 0 : (width - self->frame_width_) / 2;
                src_offsetY = (self->frame_height_ == height) ? 0 : (height - self->frame_height_) / 2;
                src_W       = self->frame_width_;
                src_H       = self->frame_height_;
                dst_offsetX = 0;
                dst_offsetY = 0;
                dst_W       = src_W;
                dst_H       = src_H;
            } else if ((self->frame_height_ >= height) && (self->frame_width_ <= width)) {
                src_offsetX = (self->frame_width_ == width) ? 0 : (width - self->frame_width_) / 2;
                src_offsetY = 0;
                src_W       = self->frame_width_;
                src_H       = height;
                dst_offsetX = 0;
                dst_offsetY = (self->frame_height_ == src_H) ? 0 : (self->frame_height_ - src_H) / 2;
                dst_W       = src_W;
                dst_H       = src_H;
            } else {
                src_offsetX = 0;
                src_offsetY = (self->frame_height_ == height) ? 0 : (height - self->frame_height_) / 2;
                src_W       = width;
                src_H       = self->frame_height_;
                dst_offsetX = (self->frame_width_ == src_W) ? 0 : (self->frame_width_ - src_W) / 2;
                dst_offsetY = 0;
                dst_W       = src_W;
                dst_H       = src_H;
            }
            cv::Mat yuv_src(height, width, CV_8UC2, pData);
            yuv_src(cv::Rect(src_offsetX, src_offsetY, src_W, src_H))
                .copyTo(self->yuv_dist_(cv::Rect(dst_offsetX, dst_offsetY, dst_W, dst_H)));
            if (self->out_callback_)
                self->out_callback_(self->yuv_dist_.data, self->frame_height_ * self->frame_width_ * 2);
        }
    }

    void set_output(task_callback_t out_callback)
    {
        out_callback_ = out_callback;
    }
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
    static bool parse_axera_config(const nlohmann::json &config_body, const nlohmann::json &file_body,
                                   void **custom_config)
    {
        std::string rtsp_config;
        static axera_config_t axera_config;
        memset(&axera_config, 0, sizeof(axera_config_t));
        if (config_body.contains("rtsp")) {
            rtsp_config = config_body.at("rtsp");
        }
        int frame_width  = config_body.at("frame_width");
        int frame_height = config_body.at("frame_height");
        *custom_config   = (void *)&axera_config;

        {
#undef CONFIG_OBJECT
#define CONFIG_OBJECT axera_config
            CONFIG_AUTO_SET_DEFAULT(file_body["cap_param"], VinParam.eSysMode, COMMON_VIN_SENSOR);
            CONFIG_AUTO_SET_DEFAULT(file_body["cap_param"], VinParam.eHdrMode, AX_SNS_LINEAR_MODE);
            CONFIG_AUTO_SET_DEFAULT(file_body["cap_param"], VinParam.bAiispEnable, AX_TRUE);
        }

        if (rtsp_config.empty() == false) {
            AX_VENC_CHN_ATTR_T stVencChnAttr;
            memset(&stVencChnAttr, 0, sizeof(AX_VENC_CHN_ATTR_T));
#undef CONFIG_OBJECT
#define CONFIG_OBJECT stVencChnAttr
            if (rtsp_config.find("h264") != std::string::npos) {
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.enType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u32MaxPicWidth);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u32MaxPicHeight);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.enMemSource);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u32BufSize);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.enProfile);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.enLevel);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.enTier);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u32PicWidthSrc);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u32PicHeightSrc);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.stCropCfg.bEnable);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.stCropCfg.stRect.s32X);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.stCropCfg.stRect.s32Y);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.stCropCfg.stRect.u32Width);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.stCropCfg.stRect.u32Height);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.enRotation);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.enLinkMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.bDeBreathEffect);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.bRefRingbuf);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.s32StopWaitTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u8InFifoDepth);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u8OutFifoDepth);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.u32SliceNum);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stVencAttr.stAttrH265e.bRcnRefShareBuf);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.enRcMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.s32FirstFrameStartQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stFrameRate.fSrcFrameRate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stFrameRate.fDstFrameRate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32BitRate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32MaxIprop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32MinIprop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Cbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32MaxBitRate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.enVQ);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264Vbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32MaxBitRate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264AVbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QVbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QVbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QVbr.u32TargetBitRate);

                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MinQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MaxQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MaxIprop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MinIprop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32MaxBitRate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32ShortTermStatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32LongTermStatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32LongTermMaxBitrate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32LongTermMinBitrate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32ExtraBitPercent);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.u32LongTermStatTimeUnit);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264CVbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264FixQp.u32Gop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264FixQp.u32IQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264FixQp.u32PQp);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264FixQp.u32BQp);

                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QpMap.u32Gop);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QpMap.u32StatTime);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QpMap.u32TargetBitRate);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QpMap.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QpMap.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QpMap.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stRcAttr.stH264QpMap.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.enGopMode);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stNormalP.stPicConfig.s32QpOffset);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stNormalP.stPicConfig.f32QpFactor);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stOneLTR.stPicConfig.s32QpOffset);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stOneLTR.stPicConfig.f32QpFactor);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stOneLTR.stPicSpecialConfig.s32QpOffset);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stOneLTR.stPicSpecialConfig.f32QpFactor);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stOneLTR.stPicSpecialConfig.s32Interval);
                CONFIG_AUTO_SET(file_body["h264_config_param"], stGopAttr.stSvcT.u32GopSize);
            } else if (rtsp_config.find("h265") != std::string::npos) {
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.enType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u32MaxPicWidth);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u32MaxPicHeight);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.enMemSource);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u32BufSize);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.enProfile);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.enLevel);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.enTier);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u32PicWidthSrc);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u32PicHeightSrc);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.stCropCfg.bEnable);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.stCropCfg.stRect.s32X);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.stCropCfg.stRect.s32Y);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.stCropCfg.stRect.u32Width);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.stCropCfg.stRect.u32Height);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.enRotation);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.enLinkMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.bDeBreathEffect);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.bRefRingbuf);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.s32StopWaitTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u8InFifoDepth);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u8OutFifoDepth);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.u32SliceNum);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stVencAttr.stAttrH265e.bRcnRefShareBuf);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.enRcMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.s32FirstFrameStartQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stFrameRate.fSrcFrameRate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stFrameRate.fDstFrameRate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32BitRate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32MaxIprop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32MinIprop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Cbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32MaxBitRate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.enVQ);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265Vbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32MaxBitRate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265AVbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QVbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QVbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QVbr.u32TargetBitRate);

                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32Gop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32StatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MaxQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MinQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MaxIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MinIQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MinQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MaxQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.s32DeBreathQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32IdrQpDeltaRange);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MaxIprop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MinIprop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32MaxBitRate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32ShortTermStatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32LongTermStatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32LongTermMaxBitrate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32LongTermMinBitrate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32ExtraBitPercent);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.u32LongTermStatTimeUnit);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.s32IntraQpDelta);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265CVbr.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265FixQp.u32Gop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265FixQp.u32IQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265FixQp.u32PQp);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265FixQp.u32BQp);

                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QpMap.u32Gop);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QpMap.u32StatTime);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QpMap.u32TargetBitRate);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QpMap.stQpmapInfo.enCtbRcMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QpMap.stQpmapInfo.enQpmapQpType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QpMap.stQpmapInfo.enQpmapBlockType);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stRcAttr.stH265QpMap.stQpmapInfo.enQpmapBlockUnit);

                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.enGopMode);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stNormalP.stPicConfig.s32QpOffset);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stNormalP.stPicConfig.f32QpFactor);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stOneLTR.stPicConfig.s32QpOffset);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stOneLTR.stPicConfig.f32QpFactor);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stOneLTR.stPicSpecialConfig.s32QpOffset);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stOneLTR.stPicSpecialConfig.f32QpFactor);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stOneLTR.stPicSpecialConfig.s32Interval);
                CONFIG_AUTO_SET(file_body["h265_config_param"], stGopAttr.stSvcT.u32GopSize);
            }

            try {
                std::regex pattern(R"(rtsp\.(\d+)[xX-](\d+)\.h(264|265))");
                std::smatch matches;
                if (std::regex_search(rtsp_config, matches, pattern)) {
                    if (matches.size() >= 3) {
                        stVencChnAttr.stVencAttr.u32PicWidthSrc  = std::stoi(matches[1].str());
                        stVencChnAttr.stVencAttr.u32PicHeightSrc = std::stoi(matches[2].str());
                    }
                }
            } catch (...) {
                return true;
            }
            if ((stVencChnAttr.stVencAttr.u32PicWidthSrc < frame_width) ||
                (stVencChnAttr.stVencAttr.u32PicHeightSrc < frame_height)) {
                return true;
            }
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
            init_rtsp(&stVencChnAttr);
#endif
        }
        return false;
    }
#endif
    bool parse_config(const nlohmann::json &config_body)
    {
        try {
            response_format_ = config_body.at("response_format");
            enoutput_        = config_body.at("enoutput");
            devname_         = config_body.at("input");
            frame_width_     = config_body.at("frame_width");
            frame_height_    = config_body.at("frame_height");
            if (config_body.contains("enable_webstream")) {
                enable_webstream_ = config_body.at("enable_webstream");
            } else {
                enable_webstream_ = false;
            }

        } catch (...) {
            return true;
        }
        enstream_  = (response_format_.find("stream") != std::string::npos);
        enjpegout_ = (response_format_.find("jpeg") != std::string::npos);
        yuv_dist_  = cv::Mat(frame_height_, frame_width_, CV_8UC2, cv::Scalar(0, 128));
        if (devname_.find("/dev/video") != std::string::npos) {
            hal_camera_open  = camera_open;
            hal_camera_close = camera_close;
            hal_parse_config = NULL;
        }
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
        else if (devname_.find("axera_") != std::string::npos) {
            hal_camera_open  = axera_camera_open;
            hal_camera_close = axera_camera_close;
            hal_parse_config = llm_task::parse_axera_config;
        }
#endif
        else {
            return true;
        }
        {
            nlohmann::json error_body;
            nlohmann::json file_body;
            std::string base_model_path;
            std::string base_model_config_path;
            std::list<std::string> config_file_paths =
                get_config_file_paths(base_model_path, base_model_config_path, "camera");
            try {
                for (auto file_name : config_file_paths) {
                    std::ifstream config_file(file_name);
                    if (!config_file.is_open()) {
                        SLOGW("config file :%s miss", file_name.c_str());
                        continue;
                    }
                    SLOGI("config file :%s read", file_name.c_str());
                    config_file >> file_body;
                    config_file.close();
                    break;
                }
                if (file_body.empty()) {
                    SLOGE("all config file miss");
                    return true;
                }
                if (hal_parse_config) {
                    if (hal_parse_config(config_body, file_body, &cam_config)) return true;
                }
            } catch (...) {
                return true;
            }
        }

        return false;
    }

    int load_model(const nlohmann::json &config_body)
    {
        if (parse_config(config_body)) {
            return -1;
        }
        try {
            cam = hal_camera_open(devname_.c_str(), frame_width_, frame_height_, 30, cam_config);
            if (cam == NULL) {
                printf("Camera open failed \n");
                return -1;
            }
            cam->ctx_ = static_cast<void *>(this);
            cam->camera_capture_callback_set(cam, on_cap_fream);
            cam->camera_capture_start(cam);
        } catch (...) {
            SLOGE("config file read false");
            return -3;
        }
        return 0;
    }

    void inference(const std::string &msg)
    {
        // std::cout << msg << std::endl;
        if (out_callback_) out_callback_("None", 4);
    }

    llm_task(const std::string &workid)
    {
        cam        = NULL;
        cam_config = NULL;
    }

    void start()
    {
    }

    void stop()
    {
        if (cam) {
            cam->camera_capture_stop(cam);
            hal_camera_close(cam);
            cam = NULL;
        }
        if (hv_tcpserver_) {
            hv_tcpserver_->stop();
            hv_tcpserver_.reset();
        }
    }

    ~llm_task()
    {
        stop();
    }
};

class llm_camera : public StackFlow {
private:
    std::unordered_map<int, std::shared_ptr<llm_task>> llm_task_;

public:
    llm_camera() : StackFlow("camera")
    {
        rpc_ctx_->register_rpc_action(
            "list_camera", std::bind(&llm_camera::list_camera, this, std::placeholders::_1, std::placeholders::_2));
    }

    std::string list_camera(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &rawdata)
    {
        auto _rawdata = rawdata->string();
        nlohmann::json req_body;
        std::string zmq_url    = rawdata->get_param(0);
        std::string param_json = rawdata->get_param(1);
        std::vector<std::string> devices;
        glob_t glob_result;
        glob("/dev/video*", GLOB_TILDE, NULL, &glob_result);
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            devices.push_back(std::string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
        send("camera.devices", devices, LLM_NO_ERROR, sample_json_str_get(param_json, "work_id"), zmq_url);
        return LLM_NONE;
    }

    void task_output(const std::weak_ptr<llm_task> llm_task_obj_weak,
                     const std::weak_ptr<llm_channel_obj> llm_channel_weak, const void *data, int size)
    {
        auto llm_task_obj = llm_task_obj_weak.lock();
        auto llm_channel  = llm_channel_weak.lock();
        if (!(llm_task_obj && llm_channel)) {
            return;
        }
        std::vector<uchar> jpeg_image;
        // StackFlow output
        std::string out_data((char *)data, size);
        llm_channel->send_raw_to_pub(out_data);
        // user output
        if (llm_task_obj->enoutput_) {
            std::string base64_data;
            if (llm_task_obj->enjpegout_) {
                cv::Mat yuv_image(llm_task_obj->frame_height_, llm_task_obj->frame_width_, CV_8UC2, (void *)data);
                cv::Mat bgr_image;
                cv::cvtColor(yuv_image, bgr_image, cv::COLOR_YUV2BGR_YUYV);
                cv::imencode(".jpg", bgr_image, jpeg_image);
                std::string in_data((char *)jpeg_image.data(), jpeg_image.size());
                StackFlows::encode_base64(in_data, base64_data);
            } else {
                StackFlows::encode_base64(out_data, base64_data);
            }
            std::string out_json_str;
            out_json_str.reserve(llm_channel->request_id_.size() + llm_channel->work_id_.size() + base64_data.size() +
                                 128);
            out_json_str += R"({"request_id":")";
            out_json_str += llm_channel->request_id_;
            out_json_str += R"(","work_id":")";
            out_json_str += llm_channel->work_id_;
            out_json_str += R"(","object":")";
            out_json_str += llm_task_obj->response_format_;
            out_json_str += R"(","error":{"code":0, "message":""},"data":")";
            out_json_str += base64_data;
            out_json_str += "\"}\n";
            llm_channel->send_raw_to_usr(out_json_str);
        }
        // webstream output
        if (llm_task_obj->enable_webstream_) {
            if (!llm_task_obj->hv_tcpserver_) {
                llm_task_obj->hv_tcpserver_ = std::make_unique<hv::TcpServer>();
                int listenfd                = llm_task_obj->hv_tcpserver_->createsocket(8989);
                if (listenfd < 0) {
                    llm_task_obj->hv_tcpserver_.reset();
                    return;
                }
                llm_task_obj->hv_tcpserver_->onConnection = [](const hv::SocketChannelPtr &channel) {
                    std::string peeraddr = channel->peeraddr();
                    if (channel->isConnected()) {
                        memset(http_response_buff, 0, 1024);
                        time_t current_time;
                        struct tm *time_info;
                        time(&current_time);
                        time_info = gmtime(&current_time);
                        char time_str[30];
                        strftime(time_str, sizeof(time_str), "%a, %d %b %Y %H:%M:%S GMT", time_info);
                        sprintf(http_response_buff, http_response, time_str);
                        channel->write(http_response_buff);
                    }
                };
                llm_task_obj->hv_tcpserver_->onMessage = [](const hv::SocketChannelPtr &channel, hv::Buffer *buf) {};
                llm_task_obj->hv_tcpserver_->setThreadNum(1);
                llm_task_obj->hv_tcpserver_->start();
            }
            llm_task_obj->hv_tcpserver_->foreachChannel([&](const hv::SocketChannelPtr &channel) {
                if (jpeg_image.empty()) {
                    cv::Mat yuv_image(llm_task_obj->frame_height_, llm_task_obj->frame_width_, CV_8UC2, (void *)data);
                    cv::Mat bgr_image;
                    cv::cvtColor(yuv_image, bgr_image, cv::COLOR_YUV2BGR_YUYV);
                    cv::imencode(".jpg", bgr_image, jpeg_image);
                }
                char tmpsdas[256];
                struct timeval tv;
                gettimeofday(&tv, NULL);
                double timestamp = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
                memset(http_response_buff1, 0, 1024);
                sprintf(http_response_buff1, http_jpeg_response, timestamp, jpeg_image.size());
                channel->write(http_response_buff1);
                channel->write(jpeg_image.data(), jpeg_image.size());
            });
        }
    }

    int setup(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        if ((llm_task_channel_.size() - 1) == MAX_TASK_NUM) {
            error_body["code"]    = -21;
            error_body["message"] = "task full";
            send("None", "None", error_body, "llm");
            return -1;
        }
        int work_id_num   = sample_get_work_id_num(work_id);
        auto llm_channel  = get_channel(work_id);
        auto llm_task_obj = std::make_shared<llm_task>(work_id);
        nlohmann::json config_body;
        try {
            config_body = nlohmann::json::parse(data);
        } catch (...) {
            error_body["code"]    = -2;
            error_body["message"] = "json format error.";
            send("None", "None", error_body, unit_name_);
            return -2;
        }
        int ret = llm_task_obj->load_model(config_body);
        if (ret == 0) {
            llm_channel->set_output(llm_task_obj->enoutput_);
            llm_channel->set_stream(llm_task_obj->enstream_);
            llm_task_obj->set_output(std::bind(&llm_camera::task_output, this, std::weak_ptr<llm_task>(llm_task_obj),
                                               std::weak_ptr<llm_channel_obj>(llm_channel), std::placeholders::_1,
                                               std::placeholders::_2));
            llm_task_[work_id_num] = llm_task_obj;
            send("None", "None", LLM_NO_ERROR, work_id);
            return 0;
        } else {
            error_body["code"]    = -5;
            error_body["message"] = "Model loading failed.";
            send("None", "None", error_body, unit_name_);
            return -1;
        }
    }

    void taskinfo(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json req_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (WORK_ID_NONE == work_id_num) {
            std::vector<std::string> task_list;
            std::transform(llm_task_channel_.begin(), llm_task_channel_.end(), std::back_inserter(task_list),
                           [](const auto task_channel) { return task_channel.second->work_id_; });
            req_body = task_list;
            send("camera.tasklist", req_body, LLM_NO_ERROR, work_id);
        } else {
            if (llm_task_.find(work_id_num) == llm_task_.end()) {
                req_body["code"]    = -6;
                req_body["message"] = "Unit Does Not Exist";
                send("None", "None", req_body, work_id);
                return;
            }
            auto llm_task_obj           = llm_task_[work_id_num];
            req_body["response_format"] = llm_task_obj->response_format_;
            req_body["enoutput"]        = llm_task_obj->enoutput_;
            req_body["input"]           = llm_task_obj->devname_;
            req_body["frame_width"]     = llm_task_obj->frame_width_;
            req_body["frame_height"]    = llm_task_obj->frame_height_;
            send("camera.taskinfo", req_body, LLM_NO_ERROR, work_id);
        }
    }

    int exit(const std::string &work_id, const std::string &object, const std::string &data) override
    {
        nlohmann::json error_body;
        int work_id_num = sample_get_work_id_num(work_id);
        if (llm_task_.find(work_id_num) == llm_task_.end()) {
            error_body["code"]    = -6;
            error_body["message"] = "Unit Does Not Exist";
            send("None", "None", error_body, work_id);
            return -1;
        }
        llm_task_[work_id_num]->stop();
        auto llm_channel = get_channel(work_id_num);
        llm_channel->stop_subscriber("");
        llm_task_.erase(work_id_num);
        send("None", "None", LLM_NO_ERROR, work_id);
        return 0;
    }

    ~llm_camera()
    {
        while (1) {
            auto iteam = llm_task_.begin();
            if (iteam == llm_task_.end()) {
                break;
            }
            iteam->second->stop();
            get_channel(iteam->first)->stop_subscriber("");
            iteam->second.reset();
            llm_task_.erase(iteam->first);
        }
    }
};

int main(int argc, char *argv[])
{
    signal(SIGTERM, __sigint);
    signal(SIGINT, __sigint);
    mkdir("/tmp/llm", 0777);
    llm_camera llm;
    while (!main_exit_flage) {
        sleep(1);
    }
    llm.llm_firework_exit();
    return 0;
}
