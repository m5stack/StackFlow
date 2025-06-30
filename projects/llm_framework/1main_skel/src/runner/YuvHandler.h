/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#ifndef _YUV_HANDLER_H_
#define _YUV_HANDLER_H_

#include "ax_global_type.h"
#include "ax_sys_api.h"

typedef enum _YUV_COLOR {
    YUV_GREEN = 0,
    YUV_RED,
    YUV_BLUE,
    YUV_PURPLE,
    YUV_DARK_GREEN,
    YUV_YELLOW,
    YUV_LIGHT_BLUE,
    YUV_LIGHT_PURPLE,
    YUV_DARK_BLACK,
    YUV_GRAY,
    YUV_WHITE,
    YUV_COLOR_MAX,
} YUV_COLOR;

typedef struct _YUV_IMAGE_T {
    AX_U8 *pImage;
    AX_U16 nWidth;
    AX_U16 nHeight;
    AX_S16 stride;
    AX_U32 nSize;
    AX_IMG_FORMAT_E eType;
} YUV_IMAGE_T;

AX_VOID DrawLine(YUV_IMAGE_T *pstImage, AX_S16 x0, AX_S16 y0, AX_S16 x1, AX_S16 y1, YUV_COLOR eColor, AX_U8 nScale);
AX_VOID DrawRect(YUV_IMAGE_T *pstImage, AX_S16 x0, AX_S16 y0, AX_U16 w, AX_U16 h, YUV_COLOR eColor);
AX_VOID DrawPoint(YUV_IMAGE_T *pstImage, AX_S16 x, AX_S16 y, AX_U8 nScale, AX_S16 x_offset, AX_S16 y_offset,
                  YUV_COLOR eColor);

#endif /* _YUV_HANDLER_H_ */
