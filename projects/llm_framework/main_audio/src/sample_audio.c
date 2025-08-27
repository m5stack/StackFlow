#include "sample_audio.h"
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
#include <signal.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include <getopt.h>
#include <tinyalsa/pcm.h>
#include <samplerate.h>
#define BUFFER_IMPLEMENTATION
#include <stdbool.h>
#include <stdint.h>
#include "libs/buffer.h"

#include <time.h>
#define CALC_FPS(tips)                                                                                           \
    {                                                                                                            \
        static int fcnt = 0;                                                                                     \
        fcnt++;                                                                                                  \
        static struct timespec ts1, ts2;                                                                         \
        clock_gettime(CLOCK_MONOTONIC, &ts2);                                                                    \
        if ((ts2.tv_sec * 1000 + ts2.tv_nsec / 1000000) - (ts1.tv_sec * 1000 + ts1.tv_nsec / 1000000) >= 1000) { \
            printf("%s => count:%d\n", tips, fcnt);                                                              \
            ts1  = ts2;                                                                                          \
            fcnt = 0;                                                                                            \
        }                                                                                                        \
    }

static AX_S32 gplayLoopExit = 0;
static AX_S32 gcapLoopExit  = 0;

int BitsToFormat(unsigned int bits, AX_AUDIO_BIT_WIDTH_E *format)
{
    switch (bits) {
        case 32:
            *format = AX_AUDIO_BIT_WIDTH_32;
            break;
        case 24:
            *format = AX_AUDIO_BIT_WIDTH_24;
            break;
        case 16:
            *format = AX_AUDIO_BIT_WIDTH_16;
            break;
        default:
            fprintf(stderr, "%u bits is not supported.\n", bits);
            return -1;
    }

    return 0;
}

int IsUpTalkVqeEnabled(const AX_AP_UPTALKVQE_ATTR_T *pstVqeAttr)
{
    int ret = ((pstVqeAttr->stAecCfg.enAecMode != AX_AEC_MODE_DISABLE) || (pstVqeAttr->stNsCfg.bNsEnable != AX_FALSE) ||
               (pstVqeAttr->stAgcCfg.bAgcEnable != AX_FALSE));
    return ret;
}

int IsDnVqeEnabled(const AX_AP_DNVQE_ATTR_T *pstVqeAttr)
{
    int ret = ((pstVqeAttr->stNsCfg.bNsEnable != AX_FALSE) || (pstVqeAttr->stAgcCfg.bAgcEnable != AX_FALSE));
    return ret;
}

AX_AUDIO_SAMPLE_CONFIG_t play_config;
void ax_play(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit, const void *data,
             int size)
{
    gplayLoopExit               = 0;
    int ret                     = 0;
    AX_AUDIO_BIT_WIDTH_E format = AX_AUDIO_BIT_WIDTH_16;
    int num_read;
    if (BitsToFormat(bit, &format)) {
        goto FCLOSE_MIX_FILE;
    }
    uint16_t num_channels           = channel;
    uint32_t sample_rate            = rate;
    long wave_data_pos              = 0;
    AX_AUDIO_BIT_WIDTH_E mix_format = AX_AUDIO_BIT_WIDTH_16;
    ret                             = AX_SYS_Init();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Init failed! Error Code:0x%X\n", ret);
        goto FCLOSE_MIX_FILE;
    }
    AX_POOL PoolId = AX_POOL_CreatePool(&play_config.stPoolConfig);
    if (PoolId == AX_INVALID_POOLID) {
        printf("AX_POOL_CreatePool failed!\n");
        goto FREE_SYS;
    }
    ret = AX_AO_Init();
    if (ret) {
        printf("AX_AO_Init failed! Error Code:0x%X\n", ret);
        goto DESTROY_POOL;
    }
    ret = AX_AO_SetPubAttr(card, device, &play_config.stAttr);
    if (ret) {
        printf("AX_AO_SetPubAttr failed! ret = %x", ret);
        goto AO_DEINIT;
    }
    if (IsDnVqeEnabled(&play_config.stVqeAttr)) {
        ret = AX_AO_SetDnVqeAttr(card, device, &play_config.stVqeAttr);
        if (ret) {
            printf("AX_AO_SetDnVqeAttr failed! ret = %x \n", ret);
            goto AO_DEINIT;
        }
    }
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
    if (play_config.stHpfAttr.bEnable) {
        ret = AX_ACODEC_TxHpfSetAttr(card, &play_config.stHpfAttr);
        if (ret) {
            printf("AX_ACODEC_TxHpfSetAttr failed! ret = %x\n", ret);
            goto AO_DEINIT;
        }
        ret = AX_ACODEC_TxHpfEnable(card);
        if (ret) {
            printf("AX_ACODEC_TxHpfEnable failed! ret = %x\n", ret);
            goto AO_DEINIT;
        }
    }
    if (play_config.stLpfAttr.bEnable) {
        ret = AX_ACODEC_TxLpfSetAttr(card, &play_config.stLpfAttr);
        if (ret) {
            printf("AX_ACODEC_TxLpfSetAttr failed! ret = %x\n", ret);
            goto DIS_HPF;
        }
        ret = AX_ACODEC_TxLpfEnable(card);
        if (ret) {
            printf("AX_ACODEC_TxLpfEnable failed! ret = %x\n", ret);
            goto DIS_HPF;
        }
    }
    if (play_config.stEqAttr.bEnable) {
        ret = AX_ACODEC_TxEqSetAttr(card, &play_config.stEqAttr);
        if (ret) {
            printf("AX_ACODEC_TxEqSetAttr failed! ret = %x\n", ret);
            goto DIS_LPF;
        }
        ret = AX_ACODEC_TxEqEnable(card);
        if (ret) {
            printf("AX_ACODEC_TxEqEnable failed! ret = %x\n", ret);
            goto DIS_LPF;
        }
    }
#endif
    ret = AX_AO_EnableDev(card, device);
    if (ret) {
        printf("AX_AO_EnableDev failed! ret = %x \n", ret);
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
        goto DIS_EQ;
#endif
        goto DIS_AO_DEVICE;
    }
    if (play_config.gResample) {
        ret = AX_AO_EnableResample(card, device, play_config.enInSampleRate);
        if (ret) {
            printf("AX_AO_EnableResample failed! ret = %x,\n", ret);
            goto DIS_AO_DEVICE;
        }
    }
    ret = AX_AO_SetVqeVolume(card, device, (AX_F64)Volume);
    if (ret) {
        printf("AX_AO_SetVqeVolume failed! ret = %x \n", ret);
        goto DIS_AO_DEVICE;
    }
    {
        buffer_t *pcmdata = buffer_map((void *)data, size);
        AX_AUDIO_FRAME_T stFrmInfo;
        memset(&stFrmInfo, 0, sizeof(stFrmInfo));
        stFrmInfo.enBitwidth  = format;
        stFrmInfo.enSoundmode = play_config.stAttr.enSoundmode;
        AX_U64 BlkSize        = 960;
        while (!gplayLoopExit) {
            stFrmInfo.u32BlkId   = AX_POOL_GetBlock(PoolId, BlkSize, NULL);
            stFrmInfo.u64VirAddr = AX_POOL_GetBlockVirAddr(stFrmInfo.u32BlkId);
            stFrmInfo.u32Len     = buffer_read_char(pcmdata, stFrmInfo.u64VirAddr, BlkSize);
            if (stFrmInfo.u32Len == 0) {
                AX_POOL_ReleaseBlock(stFrmInfo.u32BlkId);
                break;
            }
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
            ret = AX_AO_SendFrame(card, device, &stFrmInfo, NULL, 0.f, -1);
#else
            ret = AX_AO_SendFrame(card, device, &stFrmInfo, -1);
#endif
            if (ret != AX_SUCCESS) {
                AX_POOL_ReleaseBlock(stFrmInfo.u32BlkId);
                printf("AX_AO_SendFrame error, ret: %x\n", ret);
                break;
            }
            AX_POOL_ReleaseBlock(stFrmInfo.u32BlkId);
        }
        buffer_destroy(pcmdata);
    }

    if (play_config.gInstant) {
        ret = AX_AO_ClearDevBuf(card, device);
        if (ret) {
            printf("AX_AO_ClearDevBuf failed! ret = %x\n", ret);
        }
    } else {
        AX_AO_DEV_STATE_T stStatus;
        while (1) {
            ret = AX_AO_QueryDevStat(card, device, &stStatus);
            if (stStatus.u32DevBusyNum == 0) {
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
                if (play_config.gInsertSilence) {
                    break;
                } else {
                    if (stStatus.longPcmBufDelay <= 0) {
                        break;
                    }
                }
#else
                break;
#endif
            }
            usleep(10 * 1000);
        }
    }
    printf("ao success.\n");
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
DIS_EQ:
    if (play_config.stEqAttr.bEnable) {
        ret = AX_ACODEC_TxEqDisable(card);
        if (ret) {
            printf("AX_ACODEC_TxEqDisable failed! ret= %x\n", ret);
        }
    }
DIS_LPF:
    if (play_config.stLpfAttr.bEnable) {
        ret = AX_ACODEC_TxLpfDisable(card);
        if (ret) {
            printf("AX_ACODEC_TxLpfDisable failed! ret= %x\n", ret);
        }
    }
DIS_HPF:
    if (play_config.stHpfAttr.bEnable) {
        ret = AX_ACODEC_TxHpfDisable(card);
        if (ret) {
            printf("AX_ACODEC_TxHpfDisable failed! ret= %x\n", ret);
        }
    }
#endif
DIS_AO_DEVICE:
    ret = AX_AO_DisableDev(card, device);
    if (ret) {
        printf("AX_AO_DisableDev failed! ret= %x\n", ret);
        goto FREE_SYS;
    }
AO_DEINIT:
    ret = AX_AO_DeInit();
    if (AX_SUCCESS != ret) {
        printf("AX_AO_DeInit failed! Error Code:0x%X\n", ret);
    }

DESTROY_POOL:
    ret = AX_POOL_DestroyPool(PoolId);
    if (AX_SUCCESS != ret) {
        printf("AX_POOL_DestroyPool failed! Error Code:0x%X\n", ret);
    }

FREE_SYS:
    ret = AX_SYS_Deinit();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Deinit failed! Error Code:0x%X\n", ret);
        ret = -1;
    }
    gplayLoopExit = 2;
FCLOSE_MIX_FILE:
    return;
}

void ax_close_play()
{
    gplayLoopExit = 1;
}

int ax_play_status()
{
    return gplayLoopExit;
}

AX_AUDIO_SAMPLE_CONFIG_t cap_config;
void ax_cap_start(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit,
                  AUDIOCallback callback)
{
    int ret;
    AX_AUDIO_BIT_WIDTH_E format;
    unsigned int totalFrames = 0;
    gcapLoopExit             = 0;
    if (BitsToFormat(bit, &format)) {
        gcapLoopExit = 2;
        return;
    }

    ret = AX_SYS_Init();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Init failed! Error Code:0x%X\n", ret);
        gcapLoopExit = 2;
        return;
    }
    AX_POOL PoolId = AX_POOL_CreatePool(&cap_config.stPoolConfig);
    if (PoolId == AX_INVALID_POOLID) {
        printf("AX_POOL_CreatePool failed!\n");
        goto FREE_SYS;
    }

    ret = AX_AI_Init();
    if (ret) {
        printf("AX_AI_Init failed! Error Code:0x%X\n", ret);
        goto DESTROY_POOL;
    }
    ret = AX_AI_SetPubAttr(card, device, &cap_config.aistAttr);
    if (ret) {
        printf("AX_AI_SetPubAttr failed! ret = %x\n", ret);
        goto AI_DEINIT;
    }

    ret = AX_AI_AttachPool(card, device, PoolId);
    if (ret) {
        printf("AX_AI_AttachPool failed! ret = %x\n", ret);
        goto AI_DEINIT;
    }

    unsigned int outRate = rate;
    if (IsUpTalkVqeEnabled(&cap_config.aistVqeAttr)) {
        ret = AX_AI_SetUpTalkVqeAttr(card, device, &cap_config.aistVqeAttr);
        if (ret) {
            printf("AX_AI_SetUpTalkVqeAttr failed! ret = %x\n", ret);
            goto AI_DETACHPOOL;
        }
        outRate = rate;
    }
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
    if (cap_config.stHpfAttr.bEnable) {
        ret = AX_ACODEC_RxHpfSetAttr(card, &cap_config.stHpfAttr);
        if (ret) {
            printf("AX_ACODEC_RxHpfSetAttr failed! ret = %x\n", ret);
            goto AI_DETACHPOOL;
        }
        ret = AX_ACODEC_RxHpfEnable(card);
        if (ret) {
            printf("AX_ACODEC_RxHpfEnable failed! ret = %x\n", ret);
            goto AI_DETACHPOOL;
        }
    }
    if (cap_config.stLpfAttr.bEnable) {
        ret = AX_ACODEC_RxLpfSetAttr(card, &cap_config.stLpfAttr);
        if (ret) {
            printf("AX_ACODEC_RxLpfSetAttr failed! ret = %x\n", ret);
            goto DIS_HPF;
        }
        ret = AX_ACODEC_RxLpfEnable(card);
        if (ret) {
            printf("AX_ACODEC_RxLpfEnable failed! ret = %x\n", ret);
            goto DIS_HPF;
        }
    }
    if (cap_config.stEqAttr.bEnable) {
        ret = AX_ACODEC_RxEqSetAttr(card, &cap_config.stEqAttr);
        if (ret) {
            printf("AX_ACODEC_RxEqSetAttr failed! ret = %x\n", ret);
            goto DIS_LPF;
        }
        ret = AX_ACODEC_RxEqEnable(card);
        if (ret) {
            printf("AX_ACODEC_RxEqEnable failed! ret = %x\n", ret);
            goto DIS_LPF;
        }
    }
#endif
    ret = AX_AI_EnableDev(card, device);
    if (ret) {
        printf("AX_AI_EnableDev failed! ret = %x \n", ret);
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
        goto DIS_EQ;
#endif
        goto AI_DETACHPOOL;
    }

    if (cap_config.gResample) {
        ret = AX_AI_EnableResample(card, device, cap_config.enOutSampleRate);
        if (ret) {
            printf("AX_AI_EnableResample failed! ret = %x,\n", ret);
            goto DIS_AI_DEVICE;
        }
        outRate = rate;
    }

    ret = AX_AI_SetVqeVolume(card, device, Volume);
    if (ret) {
        printf("AX_AI_SetVqeVolume failed! ret = %x\n", ret);
        goto DIS_AI_DEVICE;
    }
    {
        AX_AUDIO_FRAME_T stFrame;
        AX_S32 getNumber = 0;
        while (!gcapLoopExit) {
            ret = AX_AI_GetFrame(card, device, &stFrame, -1);
            if (ret != AX_SUCCESS) {
                printf("AX_AI_GetFrame error, ret: %x\n", ret);
                break;
            }
            getNumber++;
            callback(stFrame.u64VirAddr, stFrame.u32Len);
            totalFrames += stFrame.u32Len / 2;
            ret = AX_AI_ReleaseFrame(card, device, &stFrame);
            if (ret) {
                printf("AX_AI_ReleaseFrame failed! ret=%x\n", ret);
            }
        }
    }

    printf("totalFrames: %u\n", totalFrames);
    printf("ai success.\n");

DIS_AI_DEVICE:
    ret = AX_AI_DisableDev(card, device);
    if (ret) {
        printf("AX_AI_DisableDev failed! ret= %x\n", ret);
        goto FREE_SYS;
    }
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
DIS_EQ:
    if (cap_config.stEqAttr.bEnable) {
        ret = AX_ACODEC_RxEqDisable(card);
        if (ret) {
            printf("AX_ACODEC_RxEqDisable failed! ret= %x\n", ret);
        }
    }
DIS_LPF:
    if (cap_config.stLpfAttr.bEnable) {
        ret = AX_ACODEC_RxLpfDisable(card);
        if (ret) {
            printf("AX_ACODEC_RxLpfDisable failed! ret= %x\n", ret);
        }
    }
DIS_HPF:
    if (cap_config.stHpfAttr.bEnable) {
        ret = AX_ACODEC_RxHpfDisable(card);
        if (ret) {
            printf("AX_ACODEC_RxHpfDisable failed! ret= %x\n", ret);
        }
    }
#endif
AI_DETACHPOOL:
    ret = AX_AI_DetachPool(card, device);
    if (AX_SUCCESS != ret) {
        printf("AX_AI_DetachPool failed! Error Code:0x%X\n", ret);
    }

AI_DEINIT:
    ret = AX_AI_DeInit();
    if (AX_SUCCESS != ret) {
        printf("AX_AI_DeInit failed! Error Code:0x%X\n", ret);
    }

DESTROY_POOL:
    ret = AX_POOL_DestroyPool(PoolId);
    if (AX_SUCCESS != ret) {
        printf("AX_POOL_DestroyPool failed! Error Code:0x%X\n", ret);
    }

FREE_SYS:
    ret = AX_SYS_Deinit();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Deinit failed! Error Code:0x%X\n", ret);
        ret = -1;
    }
    gcapLoopExit = 2;
}

void ax_close_cap()
{
    gcapLoopExit = 1;
}

int ax_cap_status()
{
    return gcapLoopExit;
}