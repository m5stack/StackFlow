#include "alsa_audio.h"
#include <pcm.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

static int gcapLoopExit = 0;

void alsa_cap_start(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit,
                    AUDIOCallback callback)
{
    struct pcm_config config;
    unsigned int pcm_open_flags;
    struct pcm *pcm;
    char *buffer;
    unsigned int size;
    unsigned int frames_read;
    unsigned int total_frames_read;
    unsigned int bytes_per_frame;

    memset(&config, 0, sizeof(config));
    config.channels          = channel;
    config.rate              = rate;
    config.period_size       = 1024;
    config.period_count      = 4;
    config.format            = PCM_FORMAT_S16_LE;
    config.start_threshold   = 0;
    config.stop_threshold    = 0;
    config.silence_threshold = 0;

    pcm_open_flags = PCM_IN;
    if (1) pcm_open_flags |= PCM_MMAP;

    pcm = pcm_open(card, device, pcm_open_flags, &config);
    if (!pcm || !pcm_is_ready(pcm)) {
        fprintf(stderr, "Unable to open PCM device (%s)\n", pcm_get_error(pcm));
        return;
    }

    size   = pcm_frames_to_bytes(pcm, pcm_get_buffer_size(pcm));
    buffer = malloc(size);
    if (!buffer) {
        fprintf(stderr, "Unable to allocate %u bytes\n", size);
        pcm_close(pcm);
        return;
    }

    if (1) {
        printf("Capturing sample: %u ch, %u hz, %u bit\n", channel, rate, pcm_format_to_bits(PCM_FORMAT_S16_LE));
    }

    bytes_per_frame   = pcm_frames_to_bytes(pcm, 1);
    total_frames_read = 0;
    while (!gcapLoopExit) {
        int ret = pcm_readi(pcm, buffer, pcm_get_buffer_size(pcm));
        if (ret < 0) {
            fprintf(stderr, "Error capturing samples - %d (%s)\n", errno, strerror(errno));
            break;
        }
        frames_read = ret;
        total_frames_read += frames_read;
        callback(buffer, frames_read * bytes_per_frame);
    }

    free(buffer);
    pcm_close(pcm);
}

void alsa_close_cap()
{
    gcapLoopExit = 1;
}

int alsa_cap_status()
{
    return gcapLoopExit;
}