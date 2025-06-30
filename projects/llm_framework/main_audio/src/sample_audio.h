#pragma once
#include "ax_base_type.h"
#include "ax_aenc_api.h"
#include "ax_adec_api.h"
#include "ax_audio_process.h"
#include "ax_sys_api.h"
#include "ax_ai_api.h"
#include "ax_ao_api.h"
#include "ax_global_type.h"
#if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
#include "ax_acodec_api.h"
#include "ax_aac.h"
#endif

typedef struct {
    unsigned int card;
    unsigned int device;
    float volume;
    int channel;
    int rate;
    int bit;
    AX_POOL_CONFIG_T stPoolConfig;
    union {
        AX_AO_ATTR_T stAttr;
        AX_AI_ATTR_T aistAttr;
    };
    union {
        AX_AP_DNVQE_ATTR_T stVqeAttr;
        AX_AP_UPTALKVQE_ATTR_T aistVqeAttr;
    };
    #if defined(CONFIG_AX_620E_MSP_ENABLED) || defined(CONFIG_AX_620Q_MSP_ENABLED)
    AX_ACODEC_FREQ_ATTR_T stHpfAttr;
    AX_ACODEC_FREQ_ATTR_T stLpfAttr;
    AX_ACODEC_EQ_ATTR_T stEqAttr;
    #endif
    int gResample;
    union {
        AX_AUDIO_SAMPLE_RATE_E enInSampleRate;
        AX_AUDIO_SAMPLE_RATE_E enOutSampleRate;
    };
    AX_S32 gInstant;
    AX_S32 gInsertSilence;
    AX_S32 gDbDetection;
} AX_AUDIO_SAMPLE_CONFIG_t;

extern AX_AUDIO_SAMPLE_CONFIG_t cap_config;
extern AX_AUDIO_SAMPLE_CONFIG_t play_config;

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*AUDIOCallback)(const char *data, int size);
void ax_play(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit, const void *data,
             int size);
void ax_set_play(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit);
void ax_play_continue(const void *data, int size);
void ax_close_play();
void ax_cap_start(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit,
                  AUDIOCallback callback);
void ax_close_cap();
int ax_play_status();
int ax_cap_status();

#ifdef __cplusplus
}
#endif
