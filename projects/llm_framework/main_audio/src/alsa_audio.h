#pragma once

typedef struct AlsaConfig
{
    unsigned int card;
    unsigned int device;
    float volume;
    int channel;
    int rate;
    int bit;
} AlsaConfig;

extern AlsaConfig cap_config;
extern AlsaConfig play_config;

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*AUDIOCallback)(const char *data, int size);

void alsa_cap_start(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit,
                    AUDIOCallback callback);
void alsa_close_cap();

void alsa_play(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit, const void *data,
             int size);

void alsa_close_play();
int alsa_cap_status();

#ifdef __cplusplus
}
#endif