#include "alsa_audio.h"
#include "samplerate.h"
#include <pcm.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

static struct pcm *g_play_pcm = NULL;
static int gplayLoopExit      = 1;
static int gcapLoopExit       = 1;
AlsaConfig cap_config;
AlsaConfig play_config;

void alsa_cap_start(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit,
                    AUDIOCallback callback)
{
    gcapLoopExit = 0;
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
    config.rate              = 48000;
    config.period_size       = 120;
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

    SRC_STATE *src_state = NULL;
    float *in_float = NULL, *out_float = NULL;

    int in_frames  = pcm_get_buffer_size(pcm);
    int out_frames = (int)((float)in_frames * ((float)rate / 48000.0f) + 1);
    int out_bytes  = out_frames * channel * sizeof(short);

    if (rate != 48000) {
        src_state = src_new(SRC_SINC_FASTEST, channel, NULL);
        in_float  = malloc(in_frames * channel * sizeof(float));
        out_float = malloc(out_frames * channel * sizeof(float));
        if (!src_state || !in_float || !out_float) {
            fprintf(stderr, "Unable to allocate resample buffers\n");
            free(buffer);
            if (in_float) free(in_float);
            if (out_float) free(out_float);
            if (src_state) src_delete(src_state);
            pcm_close(pcm);
            return;
        }
    }

    while (!gcapLoopExit) {
        int ret = pcm_readi(pcm, buffer, in_frames);
        if (ret < 0) {
            fprintf(stderr, "Error capturing samples - %d (%s)\n", errno, strerror(errno));
            break;
        }

        frames_read = ret;
        total_frames_read += frames_read;

        if (rate == 48000) {
            callback(buffer, frames_read * bytes_per_frame);
        } else {
            short *in_short = (short *)buffer;
            for (int i = 0; i < frames_read * channel; ++i) {
                in_float[i] = in_short[i] / 32768.0f;
            }
            SRC_DATA src_data;
            src_data.data_in       = in_float;
            src_data.input_frames  = frames_read;
            src_data.data_out      = out_float;
            src_data.output_frames = out_frames;
            src_data.src_ratio     = (double)rate / 48000.0;
            src_data.end_of_input  = 0;
            int error              = src_process(src_state, &src_data);
            if (error) {
                fprintf(stderr, "SRC error: %s\n", src_strerror(error));
                break;
            }
            short *out_short = malloc(src_data.output_frames_gen * channel * sizeof(short));
            for (int i = 0; i < src_data.output_frames_gen * channel; ++i) {
                float sample = out_float[i];
                if (sample > 1.0f) sample = 1.0f;
                if (sample < -1.0f) sample = -1.0f;
                out_short[i] = (short)(sample * 32767.0f);
            }
            callback((const char *)out_short, src_data.output_frames_gen * channel * sizeof(short));
            free(out_short);
        }
    }

    if (rate != 48000) {
        free(in_float);
        free(out_float);
        src_delete(src_state);
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

void alsa_play(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit, const void *data,
               int size)
{
    gplayLoopExit = 0;

    struct pcm_config config;
    memset(&config, 0, sizeof(config));
    config.channels          = channel;
    config.rate              = rate;
    config.period_size       = 1024;
    config.period_count      = 2;
    config.format            = PCM_FORMAT_S16_LE;
    config.silence_threshold = config.period_size * config.period_count;
    config.stop_threshold    = config.period_size * config.period_count;
    config.start_threshold   = config.period_size;

    unsigned int pcm_open_flags = PCM_OUT;

    struct pcm *pcm = pcm_open(card, device, pcm_open_flags, &config);
    if (!pcm || !pcm_is_ready(pcm)) {
        fprintf(stderr, "Unable to open PCM playback device (%s)\n", pcm ? pcm_get_error(pcm) : "invalid pcm");
        if (pcm) {
            pcm_close(pcm);
        }
        gplayLoopExit = 2;
        return;
    }

    int frames         = pcm_bytes_to_frames(pcm, size);
    int written_frames = pcm_writei(pcm, data, frames);
    if (written_frames < 0) {
        fprintf(stderr, "PCM playback error %s\n", pcm_get_error(pcm));
    }
    printf("Played %d frames\n", written_frames);
    pcm_close(pcm);
    gplayLoopExit = 2;
}

void alsa_close_play()
{
    gplayLoopExit = 1;
}

int alsa_play_status()
{
    return gplayLoopExit;
}