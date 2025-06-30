/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include "YuvHandler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

///
typedef struct _YUV_COLOR_S {
    AX_U8 Y;
    AX_U8 U;
    AX_U8 V;
} YUV_COLOR_S;

YUV_COLOR_S g_YuvColors[YUV_COLOR_MAX] = {
    {0x00, 0x00, 0x00},  // green
    {0x00, 0x00, 0xff},  // red
    {0x00, 0xff, 0x00},  // blue
    {0x00, 0xff, 0xff},  // purple
    {0xff, 0x00, 0x00},  // dark green
    {0xff, 0x00, 0xff},  // yellow
    {0xff, 0xff, 0x00},  // light blue
    {0xff, 0xff, 0xff},  // light purple
    {0x00, 0x80, 0x80},  // dark black
    {0x80, 0x80, 0x80},  // gray
    {0xff, 0x80, 0x80}   // white
};

static AX_VOID DrawPointInternal(YUV_IMAGE_T *pstImage, AX_U8 *y, AX_U8 *u, AX_U8 *v, AX_U16 x0, AX_U16 y0,
                                 YUV_COLOR eColor) {
    if (!pstImage) {
        return;
    }

    AX_S16 m_stride = pstImage->stride;
    AX_S16 m_eType = pstImage->eType;

    AX_U32 y_offset = 0;
    AX_U32 u_offset = 0;
    AX_U32 v_offset = 0;
    switch (m_eType) {
        case AX_FORMAT_YUV420_PLANAR:  // YUV420 I420
            /* YYYY...UUUU...VVVV */
            y_offset = (AX_U32)(y0 * m_stride + x0);
            u_offset = (AX_U32)((y0 / 2) * (m_stride / 2) + x0 / 2);
            v_offset = u_offset;

            y[y_offset] = g_YuvColors[eColor].Y;
            u[u_offset] = g_YuvColors[eColor].U;
            v[v_offset] = g_YuvColors[eColor].V;
            break;

        case AX_FORMAT_YUV420_SEMIPLANAR:  // YUV420SP NV12
            /* YYYY...UVUV */
            y_offset = (AX_U32)(y0 * m_stride + x0);
            u_offset = (AX_U32)((y0 / 2) * m_stride + x0 / 2 * 2);
            v_offset = u_offset + 1;

            if (g_YuvColors[eColor].Y == 0xFF) {
                y[y_offset] = g_YuvColors[eColor].Y;
            } else {
                u[u_offset] = g_YuvColors[eColor].U;
                v[v_offset] = g_YuvColors[eColor].V;
            }
            break;

        case AX_FORMAT_YUV420_SEMIPLANAR_VU:  // YUV420SP NV21
            /* YYYY...VUVU */
            y_offset = (AX_U32)(y0 * m_stride + x0);
            v_offset = (AX_U32)((y0 / 2) * m_stride + x0 / 2 * 2);
            u_offset = v_offset + 1;

            y[y_offset] = g_YuvColors[eColor].Y;
            u[u_offset] = g_YuvColors[eColor].U;
            v[v_offset] = g_YuvColors[eColor].V;
            break;

        case AX_FORMAT_YUV422_INTERLEAVED_UYVY:
            /* UYVYUYVY */
            u_offset = (AX_U32)(y0 * m_stride * 2 + x0 / 2 * 4);
            v_offset = u_offset + 2;
            y_offset = u_offset + 1;
            y[y_offset] = g_YuvColors[eColor].Y;
            y[y_offset + 2] = g_YuvColors[eColor].Y;
            y[u_offset] = g_YuvColors[eColor].U;
            y[v_offset] = g_YuvColors[eColor].V;
            break;

        case AX_FORMAT_YUV422_INTERLEAVED_YUYV:
            /* YUYVYUYV */
            y_offset = (AX_U32)(y0 * m_stride * 2 + x0 / 2 * 4);
            u_offset = y_offset + 1;
            v_offset = u_offset + 2;

            y[y_offset] = g_YuvColors[eColor].Y;
            y[y_offset + 2] = g_YuvColors[eColor].Y;
            y[u_offset] = g_YuvColors[eColor].U;
            y[v_offset] = g_YuvColors[eColor].V;
            break;

        default:
            break;
    }
}

AX_VOID DrawPoint(YUV_IMAGE_T *pstImage, AX_S16 x, AX_S16 y, AX_U8 nScale /* = 1*/, AX_S16 x_offset /* = 0*/,
                  AX_S16 y_offset /* = 0*/, YUV_COLOR eColor /* = YUV_GREEN*/) {
    if (!pstImage) {
        return;
    }

    AX_U8 *m_pImage = pstImage->pImage;
    AX_S16 m_nWidth = pstImage->nWidth;
    AX_S16 m_nHeight = pstImage->nHeight;
    AX_S16 m_stride = pstImage->stride;
    AX_S16 m_eType = pstImage->eType;

    AX_U8 *pY = NULL;
    AX_U8 *pU = NULL;
    AX_U8 *pV = NULL;
    switch (m_eType) {
        case AX_FORMAT_YUV420_PLANAR:
            pY = m_pImage;
            pU = m_pImage + m_stride * m_nHeight;
            pV = pU + m_stride * m_nHeight / 4;
            break;
        case AX_FORMAT_YUV420_SEMIPLANAR:
        case AX_FORMAT_YUV420_SEMIPLANAR_VU:
            pY = m_pImage;
            pU = m_pImage + m_stride * m_nHeight;
            pV = pU;
            break;
        case AX_FORMAT_YUV422_INTERLEAVED_UYVY:
        case AX_FORMAT_YUV422_INTERLEAVED_YUYV:
            pY = m_pImage;
            pU = pY;
            pV = pY;
            break;
        default:
            break;
    }

    AX_S16 nXStart = 0;
    AX_S16 nYStart = 0;
    for (AX_U32 hScale = 0; hScale < nScale; hScale++) {
        for (AX_U32 wScale = 0; wScale < nScale; wScale++) {
            nXStart = x * nScale + hScale - x_offset;
            nYStart = y * nScale + wScale - y_offset;
            if (nXStart < 0 || nXStart > m_nWidth) {
                break;
            }

            if (nYStart < 0 || nYStart > m_nHeight) {
                break;
            }

            DrawPointInternal(pstImage, pY, pU, pV, nXStart, nYStart, eColor);
        }
    }
}

AX_VOID DrawLine(YUV_IMAGE_T *pstImage, AX_S16 x0, AX_S16 y0, AX_S16 x1, AX_S16 y1, YUV_COLOR eColor /* = YUV_GREEN*/,
                 AX_U8 nScale /* = 1*/)

{
    if (!pstImage) {
        return;
    }

    AX_U8 *m_pImage = pstImage->pImage;
    AX_S16 m_nWidth = pstImage->nWidth;
    AX_S16 m_nHeight = pstImage->nHeight;
    AX_S16 m_stride = pstImage->stride;
    AX_S16 m_eType = pstImage->eType;

    x0 = (x0 < 0) ? 0 : x0;
    y0 = (y0 < 0) ? 0 : y0;
    x1 = (x1 < 0) ? 0 : x1;
    y1 = (y1 < 0) ? 0 : y1;

    x0 = (x0 >= m_nWidth) ? m_nWidth - 1 : x0;
    y0 = (y0 >= m_nHeight) ? m_nHeight - 1 : y0;
    x1 = (x1 >= m_nWidth) ? m_nWidth - 1 : x1;
    y1 = (y1 >= m_nHeight) ? m_nHeight - 1 : y1;

    AX_U16 dx = (x0 > x1) ? (x0 - x1) : (x1 - x0);
    AX_U16 dy = (y0 > y1) ? (y0 - y1) : (y1 - y0);

    AX_S16 xstep = (x0 < x1) ? 1 : -1;
    AX_S16 ystep = (y0 < y1) ? 1 : -1;
    AX_S16 nstep = 0, eps = 0;

    AX_U8 *pY = NULL;
    AX_U8 *pU = NULL;
    AX_U8 *pV = NULL;
    switch (m_eType) {
        case AX_FORMAT_YUV420_PLANAR:
            pY = m_pImage;
            pU = m_pImage + m_stride * m_nHeight;
            pV = pU + m_stride * m_nHeight / 4;
            break;
        case AX_FORMAT_YUV420_SEMIPLANAR:
        case AX_FORMAT_YUV420_SEMIPLANAR_VU:
            pY = m_pImage;
            pU = m_pImage + m_stride * m_nHeight;
            pV = pU;
            break;
        case AX_FORMAT_YUV422_INTERLEAVED_UYVY:
        case AX_FORMAT_YUV422_INTERLEAVED_YUYV:
            pY = m_pImage;
            pU = pY;
            pV = pY;
            break;
        default:
            break;
    }

    AX_U16 x = x0;
    AX_U16 y = y0;
    if (dx > dy) {
        while (nstep <= dx) {
            if (nScale == 1) {
                DrawPointInternal(pstImage, pY, pU, pV, x, y, eColor);
            } else {
                DrawPoint(pstImage, x, y, nScale, x * (nScale - 1), y * (nScale - 1), eColor);
            }
            eps += dy;
            if ((eps << 1) >= dx) {
                y += ystep;
                eps -= dx;
            }
            x += xstep;
            nstep++;
        }
    } else {
        while (nstep <= dy) {
            if (nScale == 1) {
                DrawPointInternal(pstImage, pY, pU, pV, x, y, eColor);
            } else {
                DrawPoint(pstImage, x, y, nScale, x * (nScale - 1), y * (nScale - 1), eColor);
            }
            eps += dx;
            if ((eps << 1) >= dy) {
                x += xstep;
                eps -= dy;
            }
            y += ystep;
            nstep++;
        }
    }
}

AX_VOID DrawRect(YUV_IMAGE_T *pstImage, AX_S16 x0, AX_S16 y0, AX_U16 w, AX_U16 h, YUV_COLOR eColor /* = YUV_GREEN*/) {
    if (w > 0 && h > 0) {
        DrawLine(pstImage, x0, y0, x0 + w, y0, eColor, 1);
        DrawLine(pstImage, x0, y0, x0, y0 + h, eColor, 1);
        DrawLine(pstImage, x0 + w, y0, x0 + w, y0 + h, eColor, 1);
        DrawLine(pstImage, x0, y0 + h, x0 + w, y0 + h, eColor, 1);
    }
}
