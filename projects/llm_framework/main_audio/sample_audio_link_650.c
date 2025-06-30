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
#include "ax_base_type.h"
#include "ax_aenc_api.h"
#include "ax_adec_api.h"
#include "ax_audio_process.h"
#include "ax_sys_api.h"
#include "wave_parser.h"
#include "ax_ai_api.h"
#include "ax_ao_api.h"
#include "ax_global_type.h"

#define FILE_NAME_SIZE 128
#define RESAMPLE_NUM   16 * 1024
// #define RESAMPLE_PERF

static unsigned int gCardNum    = 0;
static unsigned int gDeviceNum  = 0;
static unsigned int gChannels   = 2;
static unsigned int gRate       = 16000;
static unsigned int gEncodeRate = 16000;
static unsigned int gBits       = 16;
static const char *gEncoderType = "g711a";
static AX_S32 gWriteFrames      = 0;
static AX_S32 gLoopExit         = 0;
static AX_S32 gVqeSampleRate    = 16000;
static AX_S32 gResRate          = 32000;

static unsigned int gPeriodSize  = 160;
static unsigned int gPeriodCount = 4;
static int gIsWave               = 1;
static AX_AEC_CONFIG_T gAecCfg   = {
      .enAecMode = AX_AEC_MODE_DISABLE,
};
static AX_NS_CONFIG_T gNsCfg = {
    .bNsEnable             = AX_FALSE,
    .enAggressivenessLevel = 2,
};
static AX_AGC_CONFIG_T gAgcCfg = {
    .bAgcEnable     = AX_FALSE,
    .enAgcMode      = AX_AGC_MODE_FIXED_DIGITAL,
    .s16TargetLevel = -3,
    .s16Gain        = 9,
};

static AX_VAD_CONFIG_T gVadCfg = {
    .bVadEnable  = AX_FALSE,
    .u32VadLevel = 2,
};

static unsigned int gAencChannels      = 2;
static AX_AI_LAYOUT_MODE_E gLayoutMode = AX_AI_MIC_MIC;
static int gResample                   = 0;
static int gConverter                  = SRC_SINC_FASTEST;
static AX_AAC_TYPE_E gAacType          = AX_AAC_TYPE_AAC_LC;
static AX_AAC_TRANS_TYPE_E gTransType  = AX_AAC_TRANS_TYPE_ADTS;
static AX_F64 gVqeVolume               = 1.0;
static AX_S32 gGetNumber               = -1;
static AX_S32 gLoopNumber              = 1;
static const char *gInputFile          = NULL;
static const char *gOutputFile         = NULL;
static const char *gLengthFile         = NULL;
static const char *gAscFile            = NULL;
static AX_S32 gSaveFile                = 0;
static AX_S32 gCtrl                    = 0;
static AX_S32 gInstant                 = 0;
static AX_S32 gInsertSilence           = 0;
static AX_S32 gSimDrop                 = 0;
static AX_S32 gAsyncTest               = 0;
static const char *gAsyncTestName      = NULL;
static AX_S32 gAsyncTestNumber         = 10;

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
               (pstVqeAttr->stAgcCfg.bAgcEnable != AX_FALSE) || (pstVqeAttr->stVadCfg.bVadEnable != AX_FALSE));

    return ret;
}

int IsDnVqeEnabled(const AX_AP_DNVQE_ATTR_T *pstVqeAttr)
{
    int ret = ((pstVqeAttr->stNsCfg.bNsEnable != AX_FALSE) || (pstVqeAttr->stAgcCfg.bAgcEnable != AX_FALSE));

    return ret;
}

typedef struct axSAMPLE_AI_CTRL_ARGS_S {
    AI_CARD aiCardId;
    AI_DEV aiDevId;
} SAMPLE_AI_CTRL_ARGS_S;

static void *AiCtrlThread(void *arg)
{
    AX_S32 ret                        = AX_SUCCESS;
    SAMPLE_AI_CTRL_ARGS_S *aiCtrlArgs = (SAMPLE_AI_CTRL_ARGS_S *)arg;
    AX_CHAR input[10];
    AX_S32 input_length;
    AX_S32 index = 0;

    while (!gLoopExit) {
        printf(
            "please enter[save(s), unsave(a), VqeVolume=1.0(1), VqeVolume=5.0(5), EnableResample(r), "
            "DisableResample(e), ToggleResample(t), quit(q)]:\n");
        fgets(input, sizeof(input), stdin);
        input_length                = strlen(input);
        input[strcspn(input, "\n")] = 0;
        if (2 == input_length) {
            switch (input[0]) {
                case 's':
                case 'S': {
                    AX_AUDIO_SAVE_FILE_INFO_T stSaveFileInfo;
                    stSaveFileInfo.bCfg = AX_TRUE;
                    strncpy(stSaveFileInfo.aFilePath, "./", AX_MAX_AUDIO_FILE_PATH_LEN);
                    strncpy(stSaveFileInfo.aFileName, "default", AX_MAX_AUDIO_FILE_NAME_LEN);
                    stSaveFileInfo.u32FileSize = 1024;
                    ret = AX_AI_SaveFile(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId, &stSaveFileInfo);
                    printf("AX_AI_SaveFile, ret: %x\n", ret);
                    break;
                }
                case 'a':
                case 'A': {
                    AX_AUDIO_SAVE_FILE_INFO_T stSaveFileInfo;
                    stSaveFileInfo.bCfg = AX_FALSE;
                    strncpy(stSaveFileInfo.aFilePath, "./", AX_MAX_AUDIO_FILE_PATH_LEN);
                    strncpy(stSaveFileInfo.aFileName, "default", AX_MAX_AUDIO_FILE_NAME_LEN);
                    stSaveFileInfo.u32FileSize = 1024;
                    ret = AX_AI_SaveFile(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId, &stSaveFileInfo);
                    printf("AX_AI_SaveFile, ret: %x\n", ret);
                    break;
                }
                case '1': {
                    ret = AX_AI_SetVqeVolume(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId, 1.0);
                    printf("AX_AI_SetVqeVolume, ret: %x\n", ret);
                    break;
                }
                case '5': {
                    ret = AX_AI_SetVqeVolume(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId, 5.0);
                    printf("AX_AI_SetVqeVolume, ret: %x\n", ret);
                    break;
                }
                case 'r':
                case 'R': {
                    AX_AUDIO_SAMPLE_RATE_E enOutSampleRate = gResRate;
                    ret = AX_AI_EnableResample(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId, enOutSampleRate);
                    printf("AX_AI_EnableResample, ret: %x\n", ret);
                    break;
                }
                case 'e':
                case 'E': {
                    ret = AX_AI_DisableResample(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId);
                    printf("AX_AI_DisableResample, ret: %x\n", ret);
                    break;
                }
                case 't':
                case 'T': {
                    AX_AUDIO_SAMPLE_RATE_E enOutSampleRateArray[3] = {
                        AX_AUDIO_SAMPLE_RATE_8000, AX_AUDIO_SAMPLE_RATE_32000, AX_AUDIO_SAMPLE_RATE_48000};
                    ret = AX_AI_DisableResample(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId);
                    printf("AX_AI_DisableResample, ret: %x\n", ret);
                    ret = AX_AI_EnableResample(aiCtrlArgs->aiCardId, aiCtrlArgs->aiDevId, enOutSampleRateArray[index]);
                    printf("AX_AI_EnableResample, ret: %x\n", ret);
                    index++;
                    index %= 3;
                    break;
                }
                case 'q':
                case 'Q': {
                    gLoopExit = 1;
                    break;
                }
            }
        }
    }

    printf("AiCtrlThread exit\n");
    return NULL;
}

typedef struct axSAMPLE_AO_CTRL_ARGS_S {
    AO_CARD aoCardId;
    AO_DEV aoDevId;
} SAMPLE_AO_CTRL_ARGS_S;

static void *AoCtrlThread(void *arg)
{
    AX_S32 ret                        = AX_SUCCESS;
    SAMPLE_AO_CTRL_ARGS_S *aoCtrlArgs = (SAMPLE_AO_CTRL_ARGS_S *)arg;
    AX_CHAR input[10];
    AX_S32 input_length;
    AX_S32 index = 0;

    while (!gLoopExit) {
        printf(
            "please enter[save(s), unsave(a), VqeVolume=1.0(1), VqeVolume=5.0(5), EnableResample(r), "
            "DisableResample(e), ToggleResample(t), mute(m), unmute(u), PauseRecv(p), ResumeRecv(c), quit(q)]:\n");
        fgets(input, sizeof(input), stdin);
        input_length                = strlen(input);
        input[strcspn(input, "\n")] = 0;
        if (2 == input_length) {
            switch (input[0]) {
                case 's':
                case 'S': {
                    AX_AUDIO_SAVE_FILE_INFO_T stSaveFileInfo;
                    stSaveFileInfo.bCfg = AX_TRUE;
                    strncpy(stSaveFileInfo.aFilePath, "./", AX_MAX_AUDIO_FILE_PATH_LEN);
                    strncpy(stSaveFileInfo.aFileName, "default", AX_MAX_AUDIO_FILE_NAME_LEN);
                    stSaveFileInfo.u32FileSize = 1024;
                    ret = AX_AO_SaveFile(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, &stSaveFileInfo);
                    printf("AX_AO_SaveFile, ret: %x\n", ret);
                    break;
                }
                case 'a':
                case 'A': {
                    AX_AUDIO_SAVE_FILE_INFO_T stSaveFileInfo;
                    stSaveFileInfo.bCfg = AX_FALSE;
                    strncpy(stSaveFileInfo.aFilePath, "./", AX_MAX_AUDIO_FILE_PATH_LEN);
                    strncpy(stSaveFileInfo.aFileName, "default", AX_MAX_AUDIO_FILE_NAME_LEN);
                    stSaveFileInfo.u32FileSize = 1024;
                    ret = AX_AO_SaveFile(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, &stSaveFileInfo);
                    printf("AX_AO_SaveFile, ret: %x\n", ret);
                    break;
                }
                case '1': {
                    ret = AX_AO_SetVqeVolume(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, 1.0);
                    printf("AX_AO_SetVqeVolume, ret: %x\n", ret);
                    break;
                }
                case '5': {
                    ret = AX_AO_SetVqeVolume(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, 5.0);
                    printf("AX_AO_SetVqeVolume, ret: %x\n", ret);
                    break;
                }
                case 'r':
                case 'R': {
                    AX_AUDIO_SAMPLE_RATE_E enInSampleRate = gResRate;
                    ret = AX_AO_EnableResample(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, enInSampleRate);
                    printf("AX_AO_EnableResample, ret: %x\n", ret);
                    break;
                }
                case 'e':
                case 'E': {
                    ret = AX_AO_DisableResample(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId);
                    printf("AX_AO_DisableResample, ret: %x\n", ret);
                    break;
                }
                case 't':
                case 'T': {
                    AX_AUDIO_SAMPLE_RATE_E enInSampleRateArray[3] = {
                        AX_AUDIO_SAMPLE_RATE_8000, AX_AUDIO_SAMPLE_RATE_32000, AX_AUDIO_SAMPLE_RATE_48000};
                    ret = AX_AO_DisableResample(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId);
                    printf("AX_AO_DisableResample, ret: %x\n", ret);
                    ret = AX_AO_EnableResample(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, enInSampleRateArray[index]);
                    printf("AX_AO_EnableResample, ret: %x\n", ret);
                    index++;
                    index %= 3;
                    break;
                }
                case 'm':
                case 'M': {
                    AX_AUDIO_FADE_T stFade;
                    stFade.bFade         = AX_TRUE;
                    stFade.enFadeInRate  = AX_AUDIO_FADE_RATE_128;
                    stFade.enFadeOutRate = AX_AUDIO_FADE_RATE_128;
                    ret = AX_AO_SetVqeMute(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, AX_TRUE, &stFade);
                    printf("AX_AO_SetVqeMute, ret: %x\n", ret);
                    break;
                }
                case 'u':
                case 'U': {
                    AX_AUDIO_FADE_T stFade;
                    stFade.bFade         = AX_TRUE;
                    stFade.enFadeInRate  = AX_AUDIO_FADE_RATE_128;
                    stFade.enFadeOutRate = AX_AUDIO_FADE_RATE_128;
                    ret = AX_AO_SetVqeMute(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId, AX_FALSE, &stFade);
                    printf("AX_AO_SetVqeMute, ret: %x\n", ret);
                    break;
                }
                case 'p':
                case 'P': {
                    ret = AX_AO_PauseRecvFrame(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId);
                    printf("AX_AO_PauseRecvFrame, ret: %x\n", ret);
                    break;
                }
                case 'c':
                case 'C': {
                    ret = AX_AO_ResumeRecvFrame(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId);
                    printf("AX_AO_ResumeRecvFrame, ret: %x\n", ret);
                    break;
                }
                case 'q':
                case 'Q': {
                    gLoopExit = 1;
                    break;
                }
            }
        }
    }

    ret = AX_AO_DisableDev(aoCtrlArgs->aoCardId, aoCtrlArgs->aoDevId);
    if (ret) {
        printf("AX_AO_DisableDev failed! ret= %x\n", ret);
    }
    printf("AoCtrlThread exit\n");
    return NULL;
}

typedef struct axSAMPLE_AENC_ARGS_S {
    AENC_CHN aeChn;
    AX_PAYLOAD_TYPE_E payloadType;
    const char *fileExt;
} SAMPLE_AENC_ARGS_S;

static void IntToChar(AX_U32 i, unsigned char ch[4])
{
    ch[0] = i >> 24;
    ch[1] = (i >> 16) & 0xFF;
    ch[2] = (i >> 8) & 0xFF;
    ch[3] = i & 0xFF;
}

static AX_U32 CharToInt(unsigned char ch[4])
{
    return ((AX_U32)ch[0] << 24) | ((AX_U32)ch[1] << 16) | ((AX_U32)ch[2] << 8) | (AX_U32)ch[3];
}

static void *AencRecvThread(void *arg)
{
    AX_S32 ret                   = AX_SUCCESS;
    SAMPLE_AENC_ARGS_S *aencArgs = (SAMPLE_AENC_ARGS_S *)arg;
    FILE *fp_out                 = NULL;
    FILE *lengthHandle           = NULL;
    AX_U32 totalStreamBytes      = 0;
    if (gWriteFrames) {
        AX_CHAR file_path[FILE_NAME_SIZE];
        if (gOutputFile) {
            strncpy(file_path, gOutputFile, FILE_NAME_SIZE - 1);
        } else {
            snprintf(file_path, sizeof(file_path), "audio.%s", aencArgs->fileExt);
        }
        printf("Write encoded audio to: %s\n", file_path);
        fp_out = fopen(file_path, "w");
        assert(fp_out != NULL);

        if (gWriteFrames && (aencArgs->payloadType == PT_AAC) && (gTransType == AX_AAC_TRANS_TYPE_RAW)) {
            // record frame length
            char lengthfile[FILE_NAME_SIZE];
            strncpy(lengthfile, gLengthFile ? gLengthFile : "length.txt", FILE_NAME_SIZE - 1);
            lengthHandle = fopen(lengthfile, "wb");
            assert(lengthHandle != NULL);
        }
    }

    AX_S32 getNumber = 0;
    while (1) {
        AX_AUDIO_STREAM_T pstStream;
        ret = AX_AENC_GetStream(aencArgs->aeChn, &pstStream, -1);
        if (ret) {
            printf("AX_AENC_GetStream error: %x\n", ret);
            break;
        }
        getNumber++;
        totalStreamBytes += pstStream.u32Len;
        // printf("after encode, u32Seq: %u, byte_count: %u\n", pstStream.u32Seq, pstStream.u32Len);
        if (gWriteFrames && (aencArgs->payloadType == PT_OPUS)) {
            unsigned char int_field[4];
            IntToChar(pstStream.u32Len, int_field);
            if (fwrite(int_field, 1, 4, fp_out) != 4) {
                fprintf(stderr, "Error writing.\n");
            }
        }
        if (gWriteFrames && pstStream.u32Len) fwrite(pstStream.pStream, 1, pstStream.u32Len, fp_out);
        if (gWriteFrames && (aencArgs->payloadType == PT_AAC) && (gTransType == AX_AAC_TRANS_TYPE_RAW)) {
            fwrite(&(pstStream.u32Len), 1, 2, lengthHandle);
        }
        AX_AENC_ReleaseStream(aencArgs->aeChn, &pstStream);

        if (((gGetNumber > 0) && (getNumber >= gGetNumber)) || gLoopExit) {
            printf("getNumber: %d\n", getNumber);
            break;
        }
    }

    printf("totalStreamBytes: %u\n", totalStreamBytes);

    if (gWriteFrames) fclose(fp_out);
    if (gWriteFrames && (aencArgs->payloadType == PT_AAC) && (gTransType == AX_AAC_TRANS_TYPE_RAW)) {
        fclose(lengthHandle);
    }
    return NULL;
}

static int StringToPayloadTypeFileExt(const char *str, AX_PAYLOAD_TYPE_E *pType, const char **fileExt)
{
    int result = 0;
    if (!strcmp(str, "g711a")) {
        *pType   = PT_G711A;
        *fileExt = "g711a";
    } else if (!strcmp(str, "g711u")) {
        *pType   = PT_G711U;
        *fileExt = "g711u";
    } else if (!strcmp(str, "aac")) {
        *pType   = PT_AAC;
        *fileExt = "aac";
    } else if (!strcmp(str, "lpcm")) {
        *pType   = PT_LPCM;
        *fileExt = "lpcm";
    } else if (!strcmp(str, "g726")) {
        *pType   = PT_G726;
        *fileExt = "g726";
    } else if (!strcmp(str, "opus")) {
        *pType   = PT_OPUS;
        *fileExt = "opus";
    } else {
        result = -1;
    }

    return result;
}

static void *AsyncTestThread(void *arg)
{
    AX_S32 ret = AX_SUCCESS;

    if (!strcmp(gAsyncTestName, "PauseResume")) {
        sleep(1);
        for (int i = 0; i < gAsyncTestNumber; i++) {
            ret = AX_AO_PauseRecvFrame(gCardNum, gDeviceNum);
            if (ret) {
                printf("AX_AO_PauseRecvFrame failed! ret = %x", ret);
            }
            ret = AX_AO_ResumeRecvFrame(gCardNum, gDeviceNum);
            if (ret) {
                printf("AX_AO_ResumeRecvFrame failed! ret = %x", ret);
            }
        }
    } else {
        printf("Unknown async test name: %s\n", gAsyncTestName);
    }

    return NULL;
}

static void SigInt(int sigNo)
{
    printf("Catch signal %d\n", sigNo);
    gLoopExit = 1;
}

static void PrintHelp()
{
    printf("usage: sample_audio     <command> <args>\n");
    printf("commands:\n");
    printf("ai:                     ai get data.\n");
    printf("ao:                     ao play data.\n");
    printf("ai_aenc:                aenc link mode.\n");
    printf("adec_ao:                decode link mode.\n");
    printf("args:\n");
    printf("  -D:                   card number.                (support 0), default: 0\n");
    printf("  -d:                   device number.              (support 0,1,2,3), default: 0\n");
    printf("  -c:                   channels.                   (support 2,4), default: 2\n");
    printf("  -r:                   rate.                       (support 8000~48000), default: 48000\n");
    printf("  -b:                   bits.                       (support 16,32), default: 16\n");
    printf("  -p:                   period size.                (support 80~1024), default: 1024\n");
    printf("  -v:                   is wave file.               (support 0,1), default: 1\n");
    printf(
        "  -e:                   encoder type.               (support g711a, g711u, aac, lpcm, g726, opus), default: "
        "g711a\n");
    printf("  -w:                   write audio frame to file.  (support 0,1), default: 0\n");
    printf("  -G:                   get number.                 (support int), default: -1\n");
    printf("  -L:                   loop number.                (support int), default: 1\n");
    printf("  -i:                   input file.                 (support char*), default: NULL\n");
    printf("  -o:                   output file.                (support char*), default: NULL\n");
    printf("  --aec-mode:           aec mode.                   (support 0,1,2), default: 0\n");
    printf("  --sup-level:          Suppression Level.          (support 0,1,2), default: 0\n");
    printf("  --routing-mode:       routing mode.               (support 0,1,2,3,4), default: 0\n");
    printf("  --aenc-chns:          encode channels.            (support 1,2), default: 2\n");
    printf("  --layout:             layout mode.                (support 0,1,2), default: 0\n");
    printf("  --ns:                 ns enable.                  (support 0,1), default: 0\n");
    printf("  --ag-level:           aggressiveness level.       (support 0,1,2,3), default: 2\n");
    printf("  --agc:                agc enable.                 (support 0,1), default: 0\n");
    printf("  --target-level:       target level.               (support -31~0), default: -3\n");
    printf("  --gain:               compression gain.           (support 0~90), default: 9\n");
    printf("  --resample:           resample enable.            (support 0,1), default: 0\n");
    printf("  --resrate:            resample rate.              (support 8000~48000), default: 16000\n");
    printf("  --vqe-volume:         vqe volume.                 (support 0~10.0), default: 1.0\n");
    printf("  --converter:          converter type.             (support 0~4), default: 2\n");
    printf("  --aac-type:           aac type.                   (support 2,23,39), default: 2\n");
    printf("  --trans-type:         trans type.                 (support 0,2), default: 2\n");
    printf("  --asc-file:           asc file.                   (support char*), default: NULL\n");
    printf("  --length-file:        length file.                (support char*), default: NULL\n");
    printf("  --save-file:          save file.                  (support 0,1), default: 0\n");
    printf("  --ctrl:               ctrl enable.                (support 0,1), default: 0\n");
    printf("  --instant:            instant enable.             (support 0,1), default: 0\n");
    printf("  --period-count:       period count.               (support int), default: 4\n");
    printf("  --insert-silence:     insert silence enable.      (support int), default: 0\n");
    printf("  --sim-drop:           sim drop enable.            (support int), default: 0\n");
    printf("  --async-test:         async test enable.          (support int), default: 0\n");
    printf("  --async-test-name:    async test name.            (support char*), default: NULL\n");
    printf("  --async-test-number:  async test number.          (support int), default: 10\n");
    printf("  --vad:                vad enable.                 (support 0,1), default: 0\n");
    printf("  --vad-level:          vad Likelihood level.       (support 0,1,2,3), default: 2\n");
}

enum LONG_OPTION {
    LONG_OPTION_AEC_MODE = 10000,
    LONG_OPTION_SUPPRESSION_LEVEL,
    LONG_OPTION_ROUTING_MODE,
    LONG_OPTION_AENC_CHANNELS,
    LONG_OPTION_LAYOUT_MODE,
    LONG_OPTION_NS_ENABLE,
    LONG_OPTION_AGGRESSIVENESS_LEVEL,
    LONG_OPTION_AGC_ENABLE,
    LONG_OPTION_TARGET_LEVEL,
    LONG_OPTION_GAIN,
    LONG_OPTION_RESAMPLE,
    LONG_OPTION_RESAMPLE_RATE,
    LONG_OPTION_VQE_VOLUME,
    LONG_OPTION_CONVERTER,
    LONG_OPTION_AAC_TYPE,
    LONG_OPTION_AAC_TRANS_TYPE,
    LONG_OPTION_ASC_FILE,
    LONG_OPTION_LENGTH_FILE,
    LONG_OPTION_SAVE_FILE,
    LONG_OPTION_CTRL,
    LONG_OPTION_INSTANT,
    LONG_OPTION_PERIOD_COUNT,
    LONG_OPTION_INSERT_SILENCE,
    LONG_OPTION_SIM_DROP,
    LONG_OPTION_ASYNC_TEST,
    LONG_OPTION_ASYNC_TEST_NAME,
    LONG_OPTION_ASYNC_TEST_NUMBER,
    LONG_OPTION_VAD,
    LONG_OPTION_VAD_LEVEL,
    LONG_OPTION_BUTT
};

static int AudioInput()
{
    int ret             = 0;
    unsigned int card   = gCardNum;
    unsigned int device = gDeviceNum;
    AX_AUDIO_BIT_WIDTH_E format;
    unsigned int totalFrames = 0;
    FILE *output_file        = NULL;

    if (BitsToFormat(gBits, &format)) return -1;

    ret = AX_SYS_Init();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Init failed! Error Code:0x%X\n", ret);
        return -1;
    }

    AX_POOL_CONFIG_T stPoolConfig;
    stPoolConfig.MetaSize    = 8192;
    stPoolConfig.BlkSize     = 7680;
    stPoolConfig.BlkCnt      = 64;
    stPoolConfig.IsMergeMode = AX_FALSE;
    stPoolConfig.CacheMode   = POOL_CACHE_MODE_NONCACHE;
    strcpy((char *)stPoolConfig.PartitionName, "anonymous");
    AX_POOL PoolId = AX_POOL_CreatePool(&stPoolConfig);
    if (PoolId == AX_INVALID_POOLID) {
        printf("AX_POOL_CreatePool failed!\n");
        goto FREE_SYS;
    }

    AX_AUDIO_FRAME_T stFrame;
    char output_file_name[FILE_NAME_SIZE];
    if (gOutputFile) {
        strncpy(output_file_name, gOutputFile, FILE_NAME_SIZE - 1);
    } else {
        snprintf(output_file_name, FILE_NAME_SIZE, "audio.%s", gIsWave ? "wav" : "raw");
    }
    if (gWriteFrames) {
        output_file = fopen(output_file_name, "wb");
        assert(output_file != NULL);
        if (gIsWave) {
            LeaveWaveHeader(output_file);
        }
    }

    ret = AX_AI_Init();
    if (ret) {
        printf("AX_AI_Init failed! Error Code:0x%X\n", ret);
        goto DESTROY_POOL;
    }

    AX_AI_ATTR_T stAttr   = {0};
    stAttr.enBitwidth     = format;
    stAttr.enLinkMode     = AX_UNLINK_MODE;
    stAttr.enSamplerate   = gRate;
    stAttr.enLayoutMode   = gLayoutMode;
    stAttr.U32Depth       = 30;
    stAttr.u32PeriodSize  = gPeriodSize;
    stAttr.u32PeriodCount = gPeriodCount;
    stAttr.f2mic_distance = 0.06;
    ret                   = AX_AI_SetPubAttr(card, device, &stAttr);
    if (ret) {
        printf("AX_AI_SetPubAttr failed! ret = %x\n", ret);
        goto AI_DEINIT;
    }

    ret = AX_AI_AttachPool(card, device, PoolId);
    if (ret) {
        printf("AX_AI_AttachPool failed! ret = %x\n", ret);
        goto AI_DEINIT;
    }

    unsigned int outRate = gRate;
    AX_AP_UPTALKVQE_ATTR_T stVqeAttr;
    memset(&stVqeAttr, 0, sizeof(stVqeAttr));
    stVqeAttr.s32SampleRate   = gVqeSampleRate;
    stVqeAttr.u32FrameSamples = 160;
    memcpy(&stVqeAttr.stNsCfg, &gNsCfg, sizeof(AX_NS_CONFIG_T));
    memcpy(&stVqeAttr.stVadCfg, &gVadCfg, sizeof(AX_VAD_CONFIG_T));
    memcpy(&stVqeAttr.stAgcCfg, &gAgcCfg, sizeof(AX_AGC_CONFIG_T));
    memcpy(&stVqeAttr.stAecCfg, &gAecCfg, sizeof(AX_AEC_CONFIG_T));
    if (IsUpTalkVqeEnabled(&stVqeAttr)) {
        ret = AX_AI_SetUpTalkVqeAttr(card, device, &stVqeAttr);
        if (ret) {
            printf("AX_AI_SetUpTalkVqeAttr failed! ret = %x\n", ret);
            goto AI_DEINIT;
        }
        outRate = gVqeSampleRate;
    }

    ret = AX_AI_EnableDev(card, device);
    if (ret) {
        printf("AX_AI_EnableDev failed! ret = %x \n", ret);
        goto AI_DEINIT;
    }

    if (gResample) {
        AX_AUDIO_SAMPLE_RATE_E enOutSampleRate = gResRate;
        ret                                    = AX_AI_EnableResample(card, device, enOutSampleRate);
        if (ret) {
            printf("AX_AI_EnableResample failed! ret = %x,\n", ret);
            goto DIS_AI_DEVICE;
        }
        outRate = gResRate;
    }

    ret = AX_AI_SetVqeVolume(card, device, gVqeVolume);
    if (ret) {
        printf("AX_AI_SetVqeVolume failed! ret = %x\n", ret);
        goto DIS_AI_DEVICE;
    }

    if (gSaveFile) {
        AX_AUDIO_SAVE_FILE_INFO_T stSaveFileInfo;
        stSaveFileInfo.bCfg = AX_TRUE;
        strncpy(stSaveFileInfo.aFilePath, "./", AX_MAX_AUDIO_FILE_PATH_LEN);
        strncpy(stSaveFileInfo.aFileName, "default", AX_MAX_AUDIO_FILE_NAME_LEN);
        stSaveFileInfo.u32FileSize = 1024;
        ret                        = AX_AI_SaveFile(card, device, &stSaveFileInfo);
        if (ret) {
            printf("AX_AI_SaveFile failed! ret = %x\n", ret);
            goto DIS_AI_DEVICE;
        }
    }

    SAMPLE_AI_CTRL_ARGS_S aiCtrlArgs;
    aiCtrlArgs.aiCardId = card;
    aiCtrlArgs.aiDevId  = device;
    pthread_t ctrlTid;
    if (gCtrl) {
        pthread_create(&ctrlTid, NULL, AiCtrlThread, (void *)&aiCtrlArgs);
    }

    AX_S32 getNumber = 0;
    while (1) {
        ret = AX_AI_GetFrame(card, device, &stFrame, -1);
        if (ret != AX_SUCCESS) {
            printf("AX_AI_GetFrame error, ret: %x\n", ret);
            break;
        }
        getNumber++;
        if (gWriteFrames) fwrite(stFrame.u64VirAddr, 2, stFrame.u32Len / 2, output_file);

        totalFrames += stFrame.u32Len / 2;
        ret = AX_AI_ReleaseFrame(card, device, &stFrame);
        if (ret) {
            printf("AX_AI_ReleaseFrame failed! ret=%x\n", ret);
        }

        if (((gGetNumber > 0) && (getNumber >= gGetNumber)) || gLoopExit) {
            printf("getNumber: %d\n", getNumber);
            break;
        }
    }
    if (gWriteFrames) {
        if (gIsWave) {
            if ((gChannels == 2) && (stAttr.enLayoutMode != AX_AI_MIC_MIC)) {
                WriteWaveHeader(output_file, 1, outRate, gBits, totalFrames);
            } else {
                WriteWaveHeader(output_file, gChannels, outRate, gBits, totalFrames / gChannels);
            }
        }

        if (output_file) fclose(output_file);
    }

    printf("totalFrames: %u\n", totalFrames);
    printf("ai success.\n");

    if (gCtrl) {
        pthread_join(ctrlTid, NULL);
    }

DIS_AI_DEVICE:
    ret = AX_AI_DisableDev(card, device);
    if (ret) {
        printf("AX_AI_DisableDev failed! ret= %x\n", ret);
        goto FREE_SYS;
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

    return 0;
}

static int AudioOutput()
{
    int ret                     = 0;
    unsigned int card           = gCardNum;
    unsigned int device         = gDeviceNum;
    AX_AUDIO_BIT_WIDTH_E format = AX_AUDIO_BIT_WIDTH_16;
    int num_read;
    char input_file_name[FILE_NAME_SIZE];
    if (gInputFile) {
        strncpy(input_file_name, gInputFile, FILE_NAME_SIZE - 1);
    } else {
        snprintf(input_file_name, FILE_NAME_SIZE, "audio.%s", gIsWave ? "wav" : "raw");
    }
    FILE *input_file = fopen(input_file_name, "rb");
    if (input_file == NULL) {
        printf("failed to open '%s' for reading\n", input_file_name);
        return -1;
    }

    if (BitsToFormat(gBits, &format)) {
        fclose(input_file);
        return -1;
    }

    uint16_t num_channels = gChannels;
    uint32_t sample_rate  = gRate;
    long wave_data_pos    = 0;
    if (gIsWave) {
        uint16_t bits_per_sample;
        if (ParseWaveHeader(input_file_name, input_file, &num_channels, &sample_rate, &bits_per_sample)) {
            fprintf(stderr, "ParseWaveHeader '%s' error\n", input_file_name);
            fclose(input_file);
            return -1;
        }
        wave_data_pos = ftell(input_file);
        if (BitsToFormat(bits_per_sample, &format)) {
            fclose(input_file);
            return -1;
        }
    }

    ret = AX_SYS_Init();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Init failed! Error Code:0x%X\n", ret);
        return -1;
    }

    AX_POOL_CONFIG_T stPoolConfig;
    stPoolConfig.MetaSize    = 8192;
    stPoolConfig.BlkSize     = 32768;
    stPoolConfig.BlkCnt      = 64;
    stPoolConfig.IsMergeMode = AX_FALSE;
    stPoolConfig.CacheMode   = POOL_CACHE_MODE_NONCACHE;
    strcpy((char *)stPoolConfig.PartitionName, "anonymous");
    AX_POOL PoolId = AX_POOL_CreatePool(&stPoolConfig);
    if (PoolId == AX_INVALID_POOLID) {
        printf("AX_POOL_CreatePool failed!\n");
        goto FREE_SYS;
    }

    ret = AX_AO_Init();
    if (ret) {
        printf("AX_AO_Init failed! Error Code:0x%X\n", ret);
        goto DESTROY_POOL;
    }

    AX_AO_ATTR_T stAttr;
    memset(&stAttr, 0, sizeof(AX_AO_ATTR_T));
    stAttr.enBitwidth     = format;
    stAttr.enSoundmode    = (num_channels == 1 ? AX_AUDIO_SOUND_MODE_MONO : AX_AUDIO_SOUND_MODE_STEREO);
    stAttr.enLinkMode     = AX_UNLINK_MODE;
    stAttr.enSamplerate   = gRate;
    stAttr.U32Depth       = 30;
    stAttr.u32PeriodSize  = gPeriodSize;
    stAttr.u32PeriodCount = gPeriodCount;
    stAttr.bInsertSilence = gInsertSilence;
    stAttr.u32ChnCnt      = 2;
    ret                   = AX_AO_SetPubAttr(card, device, &stAttr);
    if (ret) {
        printf("AX_AI_SetPubAttr failed! ret = %x", ret);
        goto AO_DEINIT;
    }

    AX_AP_DNVQE_ATTR_T stVqeAttr;
    memset(&stVqeAttr, 0, sizeof(stVqeAttr));
    stVqeAttr.s32SampleRate   = gVqeSampleRate;
    stVqeAttr.u32FrameSamples = gPeriodSize;
    memcpy(&stVqeAttr.stNsCfg, &gNsCfg, sizeof(AX_NS_CONFIG_T));
    memcpy(&stVqeAttr.stAgcCfg, &gAgcCfg, sizeof(AX_AGC_CONFIG_T));
    if (IsDnVqeEnabled(&stVqeAttr)) {
        if (gIsWave && !gResample && sample_rate != gVqeSampleRate) {
            printf(
                "Invalid param. when you open vqe(%u) and it's not equal to insamplerate(%u), you need to set "
                "resample.\n",
                gVqeSampleRate, sample_rate);
            goto AO_DEINIT;
        }
        ret = AX_AO_SetDnVqeAttr(card, device, &stVqeAttr);
        if (ret) {
            printf("AX_AO_SetDnVqeAttr failed! ret = %x \n", ret);
            goto AO_DEINIT;
        }
    }

    ret = AX_AO_EnableDev(card, device);
    if (ret) {
        printf("AX_AO_EnableDev failed! ret = %x \n", ret);
        goto AO_DEINIT;
    }

    if (gResample) {
        if (gIsWave && sample_rate != gResRate) {
            printf("Invalid param. file's rate(%d) is not equal resample's(%d).\n", sample_rate, gResRate);
            goto DIS_AO_DEVICE;
        }
        AX_AUDIO_SAMPLE_RATE_E enInSampleRate = gResRate;
        ret                                   = AX_AO_EnableResample(card, device, enInSampleRate);
        if (ret) {
            printf("AX_AO_EnableResample failed! ret = %x,\n", ret);
            goto DIS_AO_DEVICE;
        }
    }

    if (gIsWave && !gResample && !IsDnVqeEnabled(&stVqeAttr) && sample_rate != gRate) {
        printf("Invalid param. Insamplerate(%d) is not equal to gRate(%d).\n", sample_rate, gRate);
        goto DIS_AO_DEVICE;
    }

    ret = AX_AO_SetVqeVolume(card, device, gVqeVolume);
    if (ret) {
        printf("AX_AO_SetVqeVolume failed! ret = %x \n", ret);
        goto DIS_AO_DEVICE;
    }

    if (gSaveFile) {
        AX_AUDIO_SAVE_FILE_INFO_T stSaveFileInfo;
        stSaveFileInfo.bCfg = AX_TRUE;
        strncpy(stSaveFileInfo.aFilePath, "./", AX_MAX_AUDIO_FILE_PATH_LEN);
        strncpy(stSaveFileInfo.aFileName, "default", AX_MAX_AUDIO_FILE_NAME_LEN);
        stSaveFileInfo.u32FileSize = 1024;
        ret                        = AX_AO_SaveFile(card, device, &stSaveFileInfo);
        if (ret) {
            printf("AX_AO_SaveFile failed! ret = %x\n", ret);
            goto DIS_AO_DEVICE;
        }
    }

    SAMPLE_AO_CTRL_ARGS_S aoCtrlArgs;
    aoCtrlArgs.aoCardId = card;
    aoCtrlArgs.aoDevId  = device;
    pthread_t ctrlTid;
    pthread_t asyncTestTid;
    if (gCtrl) {
        pthread_create(&ctrlTid, NULL, AoCtrlThread, (void *)&aoCtrlArgs);
    } else if (gAsyncTest && gAsyncTestName) {
        pthread_create(&asyncTestTid, NULL, AsyncTestThread, NULL);
    }

    AX_U64 BlkSize = 960;

    AX_AUDIO_FRAME_T stFrmInfo;
    memset(&stFrmInfo, 0, sizeof(stFrmInfo));
    stFrmInfo.enBitwidth  = format;
    stFrmInfo.enSoundmode = (num_channels == 1 ? AX_AUDIO_SOUND_MODE_MONO : AX_AUDIO_SOUND_MODE_STEREO);

    int sendCount     = 0;
    AX_S32 loopNumber = 0;
    do {
        stFrmInfo.u32BlkId   = AX_POOL_GetBlock(PoolId, BlkSize, NULL);
        stFrmInfo.u64VirAddr = AX_POOL_GetBlockVirAddr(stFrmInfo.u32BlkId);
        num_read             = fread(stFrmInfo.u64VirAddr, 1, BlkSize, input_file);
        if (num_read > 0) {
            sendCount++;
            stFrmInfo.u32Len = num_read;
            if (gSimDrop) {
                if (sendCount % 30 < 15) {
                    ret = AX_AO_SendFrame(card, device, &stFrmInfo, -1);
                    if (ret != AX_SUCCESS) {
                        AX_POOL_ReleaseBlock(stFrmInfo.u32BlkId);
                        printf("AX_AO_SendFrame error, ret: %x\n", ret);
                        break;
                    }
                }
                if (sendCount % 250 == 0) {
                    printf("sleep 5 second\n");
                    sleep(5);
                    printf("sleep 5 second done\n");
                }
                usleep(25 * 1000);
            } else {
                ret = AX_AO_SendFrame(card, device, &stFrmInfo, -1);
                if (ret != AX_SUCCESS) {
                    AX_POOL_ReleaseBlock(stFrmInfo.u32BlkId);
                    printf("AX_AO_SendFrame error, ret: %x\n", ret);
                    break;
                }
            }
        }
        AX_POOL_ReleaseBlock(stFrmInfo.u32BlkId);

        if (num_read <= 0) {
            loopNumber++;
            if ((gLoopNumber > 0) && (loopNumber >= gLoopNumber)) {
                printf("loopNumber: %d\n", loopNumber);
                break;
            } else {
                fseek(input_file, wave_data_pos, SEEK_SET);
            }
        }
    } while (!gLoopExit);

    if (gInstant) {
        ret = AX_AO_ClearDevBuf(card, device);
        if (ret) {
            printf("AX_AO_ClearDevBuf failed! ret = %x\n", ret);
        }
    } else {
        AX_AO_DEV_STATE_T stStatus;
        while (1) {
            ret = AX_AO_QueryDevStat(card, device, &stStatus);
            if (stStatus.u32DevBusyNum == 0) {
                break;
            }
            usleep(10 * 1000);
        }
    }
    printf("ao success.\n");

    if (gCtrl) {
        pthread_join(ctrlTid, NULL);
    } else if (gAsyncTest && gAsyncTestName) {
        pthread_join(asyncTestTid, NULL);
    }

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

    if (input_file) fclose(input_file);

    return 0;
}

static int AudioEncodeLink()
{
    int ret             = 0;
    unsigned int card   = gCardNum;
    unsigned int device = gDeviceNum;
    AX_AUDIO_BIT_WIDTH_E format;

    if (gBits != 16) {
        printf("%u bits is not supported.\n", gBits);
        return -1;
    }

    if (BitsToFormat(gBits, &format)) return -1;

    ret = AX_SYS_Init();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Init failed! Error Code:0x%X\n", ret);
        return -1;
    }

    AX_POOL_CONFIG_T stPoolConfig;
    stPoolConfig.MetaSize    = 8192;
    stPoolConfig.BlkSize     = 7680;
    stPoolConfig.BlkCnt      = 64;
    stPoolConfig.IsMergeMode = AX_FALSE;
    stPoolConfig.CacheMode   = POOL_CACHE_MODE_NONCACHE;
    strcpy((char *)stPoolConfig.PartitionName, "anonymous");
    AX_POOL PoolId = AX_POOL_CreatePool(&stPoolConfig);
    if (PoolId == AX_INVALID_POOLID) {
        printf("AX_POOL_CreatePool failed!\n");
        goto FREE_SYS;
    }

    AX_MOD_INFO_T Ai_Mod   = {AX_ID_AI, card, device};
    AX_MOD_INFO_T Aenc_Mod = {AX_ID_AENC, 0, 0};
    ret                    = AX_SYS_Link(&Ai_Mod, &Aenc_Mod);
    if (ret) {
        printf("AX_SYS_Link failed! Error Code:0x%X\n", ret);
        goto DESTROY_POOL;
    }

    ret = AX_AI_Init();
    if (ret) {
        printf("AX_AI_Init failed! Error Code:0x%X\n", ret);
        goto DESTROY_POOL;
    }

    AX_AI_ATTR_T stAttr;
    stAttr.enBitwidth     = format;
    stAttr.enLinkMode     = AX_LINK_MODE;
    stAttr.enSamplerate   = gRate;
    stAttr.enLayoutMode   = gLayoutMode;
    stAttr.U32Depth       = 30;
    stAttr.u32PeriodSize  = gPeriodSize;
    stAttr.u32PeriodCount = gPeriodCount;
    ret                   = AX_AI_SetPubAttr(card, device, &stAttr);
    if (ret) {
        printf("AX_AI_SetPubAttr failed! ret = %x\n", ret);
        goto AI_DEINIT;
    }

    ret = AX_AI_AttachPool(card, device, PoolId);
    if (ret) {
        printf("AX_AI_AttachPool failed! ret = %x", ret);
        goto AI_DEINIT;
    }

    gEncodeRate = stAttr.enSamplerate;
    AX_AP_UPTALKVQE_ATTR_T stVqeAttr;
    memset(&stVqeAttr, 0, sizeof(stVqeAttr));
    stVqeAttr.s32SampleRate   = gVqeSampleRate;
    stVqeAttr.u32FrameSamples = 160;
    memcpy(&stVqeAttr.stNsCfg, &gNsCfg, sizeof(AX_NS_CONFIG_T));
    memcpy(&stVqeAttr.stAgcCfg, &gAgcCfg, sizeof(AX_AGC_CONFIG_T));
    memcpy(&stVqeAttr.stAecCfg, &gAecCfg, sizeof(AX_AEC_CONFIG_T));
    memcpy(&stVqeAttr.stVadCfg, &gVadCfg, sizeof(AX_VAD_CONFIG_T));
    if (IsUpTalkVqeEnabled(&stVqeAttr)) {
        ret = AX_AI_SetUpTalkVqeAttr(card, device, &stVqeAttr);
        if (ret) {
            printf("AX_AI_SetUpTalkVqeAttr failed! ret = %x \n", ret);
            goto AI_DEINIT;
        }
        gEncodeRate = stVqeAttr.s32SampleRate;
    }

    ret = AX_AI_EnableDev(card, device);
    if (ret) {
        printf("AX_AI_EnableDev failed! ret = %x", ret);
        goto AI_DEINIT;
    }

    if (gResample) {
        AX_AUDIO_SAMPLE_RATE_E enOutSampleRate = gResRate;
        ret                                    = AX_AI_EnableResample(card, device, enOutSampleRate);
        if (ret) {
            printf("AX_AI_EnableResample failed! ret = %x,\n", ret);
            goto DIS_AI_DEVICE;
        }
        gEncodeRate = enOutSampleRate;
    }

    ret = AX_AI_SetVqeVolume(card, device, gVqeVolume);
    if (ret) {
        printf("AX_AI_SetVqeVolume failed! ret = %x\n", ret);
        goto DIS_AI_DEVICE;
    }

    ret = AX_AENC_Init();
    if (ret) {
        printf("AX_AENC_Init error: %x\n", ret);
        goto DIS_AI_DEVICE;
    }
    const char *fileExt;
    AX_PAYLOAD_TYPE_E payloadType;
    if (StringToPayloadTypeFileExt(gEncoderType, &payloadType, &fileExt)) {
        printf("Unknown payload type\n");
        goto DEINIT_AENC;
    }

    AENC_CHN aeChn = 0;
    AX_AENC_CHN_ATTR_T pstAttr;
    AX_AENC_AAC_ENCODER_ATTR_T aacEncoderAttr = {
        .enAacType        = gAacType,
        .enTransType      = gTransType,
        .enChnMode        = (gAencChannels == 1) ? AX_AAC_CHANNEL_MODE_1 : AX_AAC_CHANNEL_MODE_2,
        .u32GranuleLength = (gAacType == AX_AAC_TYPE_AAC_LC) ? 1024 : 480,
        .u32SampleRate    = gEncodeRate,
        .u32BitRate       = 48000};

    AX_AENC_G726_ENCODER_ATTR_T stG726EncoderAttr = {.u32BitRate = 32000};

    AX_AENC_OPUS_ENCODER_ATTR_T opusEncoderAttr = {.enApplication    = AX_OPUS_APPLICATION_RESTRICTED_LOWDELAY,
                                                   .u32SamplingRate  = gRate,
                                                   .s32Channels      = gAencChannels,
                                                   .s32BitrateBps    = 32000,
                                                   .f32FramesizeInMs = 10.0};

    pstAttr.enType     = payloadType;
    pstAttr.u32BufSize = 8;
    pstAttr.enLinkMode = AX_LINK_MODE;
    if (payloadType == PT_AAC) {
        if (gAacType == AX_AAC_TYPE_AAC_LC) {
            pstAttr.u32PtNumPerFrm = 1024;
        } else {
            pstAttr.u32PtNumPerFrm = 480;
        }
        pstAttr.pValue = &aacEncoderAttr;
    } else if (payloadType == PT_G726) {
        pstAttr.u32PtNumPerFrm = 480;
        pstAttr.pValue         = &stG726EncoderAttr;
    } else if (payloadType == PT_OPUS) {
        pstAttr.u32PtNumPerFrm = (AX_U32)((AX_F32)gRate * (opusEncoderAttr.f32FramesizeInMs / 1000.0));
        pstAttr.pValue         = &opusEncoderAttr;
    } else {
        pstAttr.u32PtNumPerFrm = 1024;
        pstAttr.pValue         = NULL;
    }

    ret = AX_AENC_CreateChn(aeChn, &pstAttr);
    if (ret) {
        printf("AX_AENC_CreateChn error: %x\n", ret);
        goto DEINIT_AENC;
    }

    if ((payloadType == PT_AAC) && (gTransType == AX_AAC_TRANS_TYPE_RAW)) {
        FILE *ascHandle = NULL;
        // record Audio Specific Config
        char ascfile[FILE_NAME_SIZE];
        strncpy(ascfile, gAscFile ? gAscFile : "asc.txt", FILE_NAME_SIZE - 1);
        ascHandle = fopen(ascfile, "wb");
        assert(ascHandle != NULL);
        fwrite(aacEncoderAttr.u8ConfBuf, 1, 64, ascHandle);
        fclose(ascHandle);
    }

    SAMPLE_AENC_ARGS_S aencArgs;
    aencArgs.aeChn       = aeChn;
    aencArgs.payloadType = payloadType;
    aencArgs.fileExt     = fileExt;
    pthread_t recvTid;
    pthread_create(&recvTid, NULL, AencRecvThread, (void *)&aencArgs);

    pthread_join(recvTid, NULL);

    printf("ai_aenc success.\n");
    ret = AX_AENC_DestroyChn(aeChn);
    if (ret) {
        printf("AX_AENC_DestroyChn failed! ret= %x\n", ret);
    }
DEINIT_AENC:
    ret = AX_AENC_DeInit();
    if (ret) {
        printf("AX_AENC_DeInit failed! ret= %x\n", ret);
    }

DIS_AI_DEVICE:
    ret = AX_AI_DisableDev(card, device);
    if (ret) {
        printf("AX_AI_DisableDev failed! ret= %x\n", ret);
        goto FREE_SYS;
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
    }

    return 0;
}

static int AudioDecodeLink()
{
    int ret             = 0;
    unsigned int card   = gCardNum;
    unsigned int device = gDeviceNum;
    FILE *input_file    = NULL;

    if (gBits != 16) {
        printf("%u bits is not supported.\n", gBits);
        return -1;
    }

    ret = AX_SYS_Init();
    if (AX_SUCCESS != ret) {
        printf("AX_SYS_Init failed! Error Code:0x%X\n", ret);
        return -1;
    }

    AX_POOL_CONFIG_T stPoolConfig;
    stPoolConfig.MetaSize    = 8192;
    stPoolConfig.BlkSize     = 384000;
    stPoolConfig.BlkCnt      = 64;
    stPoolConfig.IsMergeMode = AX_FALSE;
    stPoolConfig.CacheMode   = POOL_CACHE_MODE_NONCACHE;
    strcpy((char *)stPoolConfig.PartitionName, "anonymous");
    AX_POOL PoolId = AX_POOL_CreatePool(&stPoolConfig);
    if (PoolId == AX_INVALID_POOLID) {
        printf("AX_POOL_CreatePool failed!\n");
        goto FREE_SYS;
    }

    AX_MOD_INFO_T Ao_Mod   = {AX_ID_AO, card, device};
    AX_MOD_INFO_T Adec_Mod = {AX_ID_ADEC, 0, 0};
    ret                    = AX_SYS_Link(&Adec_Mod, &Ao_Mod);
    if (ret) {
        printf("AX_SYS_Link fail!!Error ret:%x \n", ret);
        goto DESTROY_POOL;
    }

    ret = AX_AO_Init();
    if (ret) {
        printf("AX_AO_Init failed! Error Code:0x%X\n", ret);
        goto DESTROY_POOL;
    }

    AX_AO_ATTR_T stAttr;
    stAttr.enBitwidth     = AX_AUDIO_BIT_WIDTH_16;
    stAttr.enLinkMode     = AX_LINK_MODE;
    stAttr.enSamplerate   = gRate;
    stAttr.U32Depth       = 30;
    stAttr.u32PeriodSize  = gPeriodSize;
    stAttr.u32PeriodCount = gPeriodCount;
    stAttr.bInsertSilence = gInsertSilence;
    stAttr.u32ChnCnt      = 2;
    ret                   = AX_AO_SetPubAttr(card, device, &stAttr);
    if (ret) {
        printf("AX_AI_SetPubAttr failed! ret = %x", ret);
        goto AO_DEINIT;
    }

    AX_AP_DNVQE_ATTR_T stVqeAttr;
    memset(&stVqeAttr, 0, sizeof(stVqeAttr));
    stVqeAttr.s32SampleRate   = gVqeSampleRate;
    stVqeAttr.u32FrameSamples = gPeriodSize;
    memcpy(&stVqeAttr.stNsCfg, &gNsCfg, sizeof(AX_NS_CONFIG_T));
    memcpy(&stVqeAttr.stAgcCfg, &gAgcCfg, sizeof(AX_AGC_CONFIG_T));
    if (IsDnVqeEnabled(&stVqeAttr)) {
        ret = AX_AO_SetDnVqeAttr(card, device, &stVqeAttr);
        if (ret) {
            printf("AX_AO_SetDnVqeAttr failed! ret = %x \n", ret);
            goto AO_DEINIT;
        }
    }

    ret = AX_AO_EnableDev(card, device);
    if (ret) {
        printf("AX_AO_EnableDev failed! ret = %x \n", ret);
        goto AO_DEINIT;
    }

    if (gResample) {
        AX_AUDIO_SAMPLE_RATE_E enInSampleRate = gResRate;
        ret                                   = AX_AO_EnableResample(card, device, enInSampleRate);
        if (ret) {
            printf("AX_AO_EnableResample failed! ret = %x,\n", ret);
            goto DIS_AO_DEVICE;
        }
    }

    ret = AX_AO_SetVqeVolume(card, device, gVqeVolume);
    if (ret) {
        printf("AX_AO_SetVqeVolume failed! ret = %x \n", ret);
        goto DIS_AO_DEVICE;
    }

    ret = AX_ADEC_Init();
    if (ret) {
        printf("AX_ADEC_Init error: %x\n", ret);
        goto DIS_AO_DEVICE;
    }
    const char *fileExt;
    AX_PAYLOAD_TYPE_E payloadType;
    if (StringToPayloadTypeFileExt(gEncoderType, &payloadType, &fileExt)) {
        printf("Unknown payload type\n");
        goto DEINIT_ADEC;
    }

    FILE *lengthHandle = NULL;
    AX_U8 raw_conf[64] = {0x0};
    AX_U8 *conf[]      = {raw_conf};
    AX_U32 conf_len    = sizeof(raw_conf);
    if ((payloadType == PT_AAC) && (gTransType == AX_AAC_TRANS_TYPE_RAW)) {
        char lengthfile[FILE_NAME_SIZE];
        strncpy(lengthfile, gLengthFile ? gLengthFile : "length.txt", FILE_NAME_SIZE - 1);
        lengthHandle = fopen(lengthfile, "rb");
        if (!lengthHandle) {
            perror(lengthfile);
            goto DEINIT_ADEC;
        }
        FILE *ascHandle = NULL;
        // record Audio Specific Config
        char ascfile[FILE_NAME_SIZE];
        strncpy(ascfile, gAscFile ? gAscFile : "asc.txt", FILE_NAME_SIZE - 1);
        ascHandle = fopen(ascfile, "rb");
        if (!ascHandle) {
            perror(ascfile);
            fclose(lengthHandle);
            goto DEINIT_ADEC;
        }
        int al = fread(raw_conf, 1, 64, ascHandle);
        if (al < 64) {
            printf("fread error\n");
            fclose(ascHandle);
            fclose(lengthHandle);
            goto DEINIT_ADEC;
        }
        fclose(ascHandle);
    }

    AX_ADEC_AAC_DECODER_ATTR_T aacDecoderAttr = {.enTransType = gTransType, .u8Conf = conf, .u32ConfLen = conf_len};

    AX_ADEC_G726_DECODER_ATTR_T stG726DecoderAttr = {.u32BitRate = 32000};

    AX_ADEC_OPUS_DECODER_ATTR_T opusDecoderAttr = {.u32SamplingRate = gRate, .s32Channels = gAencChannels};

    ADEC_CHN adChn = 0;
    AX_ADEC_CHN_ATTR_T pstAttr;
    pstAttr.enType     = payloadType;
    pstAttr.u32BufSize = 8;
    pstAttr.enLinkMode = AX_LINK_MODE;
    switch (payloadType) {
        case PT_AAC:
            pstAttr.pValue = &aacDecoderAttr;
            break;
        case PT_G726:
            pstAttr.pValue = &stG726DecoderAttr;
            break;
        case PT_OPUS:
            pstAttr.pValue = &opusDecoderAttr;
            break;
        default:
            pstAttr.pValue = NULL;
            break;
    }
    ret = AX_ADEC_CreateChn(adChn, &pstAttr);
    if (ret) {
        printf("AX_ADEC_CreateChn error: %x\n", ret);
        goto DEINIT_ADEC;
    }

    ret = AX_ADEC_AttachPool(adChn, PoolId);
    if (ret) {
        printf("AX_ADEC_AttachPool error: %x\n", ret);
        goto DEINIT_ADEC;
    }

    char *buffer;
    int size;
    int num_read;
    char file_path[FILE_NAME_SIZE];
    if (gInputFile) {
        strncpy(file_path, gInputFile, FILE_NAME_SIZE - 1);
    } else {
        snprintf(file_path, FILE_NAME_SIZE, "audio.%s", fileExt);
    }
    input_file = fopen(file_path, "rb");
    if (input_file == NULL) {
        printf("failed to open '%s' for reading", file_path);
        return 0;
    }
    size    = 960;
    int seq = 1;
    buffer  = malloc(size);
    if (!buffer) {
        fprintf(stderr, "unable to allocate %d bytes\n", size);
        return -1;
    }

    SAMPLE_AO_CTRL_ARGS_S aoCtrlArgs;
    aoCtrlArgs.aoCardId = card;
    aoCtrlArgs.aoDevId  = device;
    pthread_t ctrlTid;
    if (gCtrl) {
        pthread_create(&ctrlTid, NULL, AoCtrlThread, (void *)&aoCtrlArgs);
    }

    AX_S32 loopNumber = 0;
    if ((pstAttr.enType == PT_G711A) || (pstAttr.enType == PT_G711U) || (pstAttr.enType == PT_LPCM) ||
        (pstAttr.enType == PT_G726)) {
        do {
            num_read = fread(buffer, 1, size, input_file);
            if (num_read > 0) {
                AX_AUDIO_STREAM_T stStream;
                stStream.pStream    = (AX_U8 *)buffer;
                stStream.u64PhyAddr = 0;
                stStream.u32Len     = num_read;
                stStream.u32Seq     = seq++;
                stStream.bEof       = AX_FALSE;
                ret                 = AX_ADEC_SendStream(adChn, &stStream, AX_TRUE);
                if (ret != AX_SUCCESS) {
                    printf("AX_ADEC_SendStream error: %x\n", ret);
                    break;
                }
            } else {
                loopNumber++;
                if ((gLoopNumber > 0) && (loopNumber >= gLoopNumber)) {
                    printf("loopNumber: %d\n", loopNumber);
                    break;
                } else {
                    fseek(input_file, 0, SEEK_SET);
                }
            }
        } while (!gLoopExit);
    } else if (pstAttr.enType == PT_AAC) {
        while (!gLoopExit) {
            AX_U8 packet[10240];
            int n;
            AX_U32 packet_size;
            if (gTransType == AX_AAC_TRANS_TYPE_ADTS) {
                n = fread(packet, 1, 7, input_file);
                if (n != 7) {
                    loopNumber++;
                    if ((gLoopNumber > 0) && (loopNumber >= gLoopNumber)) {
                        printf("loopNumber: %d\n", loopNumber);
                        break;
                    } else {
                        fseek(input_file, 0, SEEK_SET);
                        continue;
                    }
                }
                if (packet[0] != 0xff || (packet[1] & 0xf0) != 0xf0) {
                    fprintf(stderr, "Not an ADTS packet\n");
                    break;
                }
                packet_size = ((packet[3] & 0x03) << 11) | (packet[4] << 3) | (packet[5] >> 5);
                n           = fread(packet + 7, 1, packet_size - 7, input_file);
                if (n != packet_size - 7) {
                    fprintf(stderr, "Partial packet\n");
                    break;
                }
            } else if (gTransType == AX_AAC_TRANS_TYPE_RAW) {
                int ll = fread(&packet_size, 1, 2, lengthHandle);
                if (ll < 1) {
                    loopNumber++;
                    if ((gLoopNumber > 0) && (loopNumber >= gLoopNumber)) {
                        printf("loopNumber: %d\n", loopNumber);
                        break;
                    } else {
                        fseek(input_file, 0, SEEK_SET);
                        fseek(lengthHandle, 0, SEEK_SET);
                        continue;
                    }
                }
                // printf("packet_size %d\n", packet_size);
                n = fread(packet, 1, packet_size, input_file);
                if (n != packet_size) {
                    fprintf(stderr, "Partial packet\n");
                    break;
                }
            } else {
                fprintf(stderr, "unsupport trans type\n");
                break;
            }

            AX_AUDIO_STREAM_T stStream;
            stStream.pStream    = (AX_U8 *)packet;
            stStream.u64PhyAddr = 0;
            stStream.u32Len     = packet_size;
            stStream.u32Seq     = seq++;
            stStream.bEof       = AX_FALSE;
            ret                 = AX_ADEC_SendStream(adChn, &stStream, AX_TRUE);
            if (ret != AX_SUCCESS) {
                printf("AX_ADEC_SendStream error: %x\n", ret);
                break;
            }
        }
    } else if (pstAttr.enType == PT_OPUS) {
        while (!gLoopExit) {
            unsigned char data[1500];
            unsigned char ch[4];
            num_read = fread(ch, 1, 4, input_file);
            if (num_read != 4) {
                loopNumber++;
                if ((gLoopNumber > 0) && (loopNumber >= gLoopNumber)) {
                    printf("loopNumber: %d\n", loopNumber);
                    break;
                } else {
                    fseek(input_file, 0, SEEK_SET);
                    continue;
                }
            }
            int len  = CharToInt(ch);
            num_read = fread(data, 1, len, input_file);
            if (num_read != (size_t)len) {
                fprintf(stderr,
                        "Ran out of input, "
                        "expecting %d bytes got %d\n",
                        len, (int)num_read);
                break;
            }

            AX_AUDIO_STREAM_T stStream;
            stStream.pStream    = (AX_U8 *)data;
            stStream.u64PhyAddr = 0;
            stStream.u32Len     = num_read;
            stStream.u32Seq     = seq++;
            stStream.bEof       = AX_FALSE;
            ret                 = AX_ADEC_SendStream(adChn, &stStream, AX_TRUE);
            if (ret != AX_SUCCESS) {
                printf("AX_ADEC_SendStream error: %x\n", ret);
                break;
            }
        }
    } else {
        printf("unsupport payload type\n");
    }

    if (gInstant) {
        ret = AX_AO_ClearDevBuf(card, device);
        if (ret) {
            printf("AX_AO_ClearDevBuf failed! ret = %x\n", ret);
        }
    } else {
        AX_AO_DEV_STATE_T stStatus;
        while (1) {
            ret = AX_AO_QueryDevStat(card, device, &stStatus);
            if (stStatus.u32DevBusyNum == 0) {
                break;
            }
            usleep(10 * 1000);
        }
    }

    free(buffer);

    printf("adec_ao success.\n");

    if (gCtrl) {
        pthread_join(ctrlTid, NULL);
    }

    ret = AX_ADEC_DestroyChn(adChn);
    if (ret) {
        printf("AX_ADEC_DestroyChn failed!ret = %x \n", ret);
    }
DEINIT_ADEC:
    ret = AX_ADEC_DeInit();
    if (ret) {
        printf("AX_ADEC_DeInit failed! ret = %x \n", ret);
    }

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

    if (input_file) fclose(input_file);

    return 0;
}

int main(int argc, char *argv[])
{
    extern int optind;
    AX_S32 c;
    AX_S32 isExit = 0;
    signal(SIGINT, SigInt);

    while (1) {
        int option_index                    = 0;
        static struct option long_options[] = {
            {"aec-mode", required_argument, 0, LONG_OPTION_AEC_MODE},
            {"sup-level", required_argument, 0, LONG_OPTION_SUPPRESSION_LEVEL},
            {"routing-mode", required_argument, 0, LONG_OPTION_ROUTING_MODE},
            {"aenc-chns", required_argument, 0, LONG_OPTION_AENC_CHANNELS},
            {"layout", required_argument, 0, LONG_OPTION_LAYOUT_MODE},
            {"ns", required_argument, 0, LONG_OPTION_NS_ENABLE},
            {"ag-level", required_argument, 0, LONG_OPTION_AGGRESSIVENESS_LEVEL},
            {"agc", required_argument, 0, LONG_OPTION_AGC_ENABLE},
            {"target-level", required_argument, 0, LONG_OPTION_TARGET_LEVEL},
            {"gain", required_argument, 0, LONG_OPTION_GAIN},
            {"resample", required_argument, 0, LONG_OPTION_RESAMPLE},
            {"resrate", required_argument, 0, LONG_OPTION_RESAMPLE_RATE},
            {"vqe-volume", required_argument, 0, LONG_OPTION_VQE_VOLUME},
            {"converter", required_argument, 0, LONG_OPTION_CONVERTER},
            {"aac-type", required_argument, 0, LONG_OPTION_AAC_TYPE},
            {"trans-type", required_argument, 0, LONG_OPTION_AAC_TRANS_TYPE},
            {"asc-file", required_argument, 0, LONG_OPTION_ASC_FILE},
            {"length-file", required_argument, 0, LONG_OPTION_LENGTH_FILE},
            {"save-file", required_argument, 0, LONG_OPTION_SAVE_FILE},
            {"ctrl", required_argument, 0, LONG_OPTION_CTRL},
            {"instant", required_argument, 0, LONG_OPTION_INSTANT},
            {"period-count", required_argument, 0, LONG_OPTION_PERIOD_COUNT},
            {"insert-silence", required_argument, 0, LONG_OPTION_INSERT_SILENCE},
            {"sim-drop", required_argument, 0, LONG_OPTION_SIM_DROP},
            {"async-test", required_argument, 0, LONG_OPTION_ASYNC_TEST},
            {"async-test-name", required_argument, 0, LONG_OPTION_ASYNC_TEST_NAME},
            {"async-test-number", required_argument, 0, LONG_OPTION_ASYNC_TEST_NUMBER},
            {"vad", required_argument, 0, LONG_OPTION_VAD},
            {"vad-level", required_argument, 0, LONG_OPTION_VAD_LEVEL},
            {0, 0, 0, 0}};

        c = getopt_long(argc, argv, "D:d:c:r:b:p:v:e:w:G:L:i:o:h", long_options, &option_index);
        if (c == -1) break;

        switch (c) {
            case 'D':
                gCardNum = atoi(optarg);
                break;
            case 'd':
                gDeviceNum = atoi(optarg);
                break;
            case 'c':
                gChannels = atoi(optarg);
                break;
            case 'r':
                gRate = atoi(optarg);
                break;
            case 'b':
                gBits = atoi(optarg);
                break;
            case 'p':
                gPeriodSize = atoi(optarg);
                break;
            case 'v':
                gIsWave = atoi(optarg);
                break;
            case 'e':
                gEncoderType = optarg;
                break;
            case 'w':
                gWriteFrames = atoi(optarg);
                break;
            case 'G':
                gGetNumber = atoi(optarg);
                break;
            case 'L':
                gLoopNumber = atoi(optarg);
                break;
            case 'i':
                gInputFile = optarg;
                break;
            case 'o':
                gOutputFile = optarg;
                break;
            case 'h':
                isExit = 1;
                break;
            case LONG_OPTION_AEC_MODE:
                gAecCfg.enAecMode = atoi(optarg);
                break;
            case LONG_OPTION_SUPPRESSION_LEVEL:
                gAecCfg.stAecFloatCfg.enSuppressionLevel = atoi(optarg);
                break;
            case LONG_OPTION_ROUTING_MODE:
                gAecCfg.stAecFixedCfg.eRoutingMode = atoi(optarg);
                break;
            case LONG_OPTION_AENC_CHANNELS:
                gAencChannels = atoi(optarg);
                break;
            case LONG_OPTION_LAYOUT_MODE:
                gLayoutMode = atoi(optarg);
                break;
            case LONG_OPTION_NS_ENABLE:
                gNsCfg.bNsEnable = atoi(optarg);
                break;
            case LONG_OPTION_AGGRESSIVENESS_LEVEL:
                gNsCfg.enAggressivenessLevel = atoi(optarg);
                break;
            case LONG_OPTION_AGC_ENABLE:
                gAgcCfg.bAgcEnable = atoi(optarg);
                break;
            case LONG_OPTION_TARGET_LEVEL:
                gAgcCfg.s16TargetLevel = atoi(optarg);
                break;
            case LONG_OPTION_GAIN:
                gAgcCfg.s16Gain = atoi(optarg);
                break;
            case LONG_OPTION_RESAMPLE:
                gResample = atoi(optarg);
                break;
            case LONG_OPTION_RESAMPLE_RATE:
                gResRate = atoi(optarg);
                break;
            case LONG_OPTION_VQE_VOLUME:
                gVqeVolume = atof(optarg);
                break;
            case LONG_OPTION_CONVERTER:
                gConverter = atoi(optarg);
                break;
            case LONG_OPTION_AAC_TYPE:
                gAacType = atoi(optarg);
                break;
            case LONG_OPTION_AAC_TRANS_TYPE:
                gTransType = atoi(optarg);
                break;
            case LONG_OPTION_ASC_FILE:
                gAscFile = optarg;
                break;
            case LONG_OPTION_LENGTH_FILE:
                gLengthFile = optarg;
                break;
            case LONG_OPTION_SAVE_FILE:
                gSaveFile = atoi(optarg);
                break;
            case LONG_OPTION_CTRL:
                gCtrl = atoi(optarg);
                break;
            case LONG_OPTION_INSTANT:
                gInstant = atoi(optarg);
                break;
            case LONG_OPTION_PERIOD_COUNT:
                gPeriodCount = atoi(optarg);
                break;
            case LONG_OPTION_INSERT_SILENCE:
                gInsertSilence = atoi(optarg);
                break;
            case LONG_OPTION_SIM_DROP:
                gSimDrop = atoi(optarg);
                break;
            case LONG_OPTION_ASYNC_TEST:
                gAsyncTest = atoi(optarg);
                break;
            case LONG_OPTION_ASYNC_TEST_NAME:
                gAsyncTestName = optarg;
                break;
            case LONG_OPTION_ASYNC_TEST_NUMBER:
                gAsyncTestNumber = atoi(optarg);
                break;
            case LONG_OPTION_VAD:
                gVadCfg.bVadEnable = atoi(optarg);
                break;
            case LONG_OPTION_VAD_LEVEL:
                gVadCfg.u32VadLevel = atoi(optarg);
                break;
            default:
                isExit = 1;
                break;
        }
    }
    if (isExit || optind >= argc) {
        PrintHelp();
        exit(0);
    }

    if (!strncmp(argv[optind], "ai_aenc", 7)) {
        AudioEncodeLink();
    } else if (!strncmp(argv[optind], "adec_ao", 7)) {
        AudioDecodeLink();
    } else if (!strncmp(argv[optind], "ai", 2)) {
        AudioInput();
    } else if (!strncmp(argv[optind], "ao", 2)) {
        AudioOutput();
    } else {
        printf("Unknown command: %s\n", argv[optind]);
    }

    return 0;
}
