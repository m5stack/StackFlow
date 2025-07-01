#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*AUDIOCallback)(const char *data, int size);

void alsa_cap_start(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit,
                    AUDIOCallback callback);
void alsa_close_cap();

#ifdef __cplusplus
}
#endif