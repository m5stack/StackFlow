#include "alsa_audio.h"
#include <pcm.h>
#include <samplerate.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static atomic_int gplayLoopExit = 2;
static atomic_int gcapLoopExit  = 2;

static struct pcm *g_play_pcm     = NULL;
static struct pcm *g_cap_pcm      = NULL;
static pthread_mutex_t g_play_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_cap_mtx  = PTHREAD_MUTEX_INITIALIZER;

AlsaConfig cap_config;
AlsaConfig play_config;

#ifndef HAVE_PCM_STOP
#define HAVE_PCM_STOP 1
#endif

static void set_play_pcm(struct pcm *pcm)
{
    pthread_mutex_lock(&g_play_mtx);
    g_play_pcm = pcm;
    pthread_mutex_unlock(&g_play_mtx);
}

static struct pcm *get_play_pcm(void)
{
    pthread_mutex_lock(&g_play_mtx);
    struct pcm *pcm = g_play_pcm;
    pthread_mutex_unlock(&g_play_mtx);
    return pcm;
}

static void clear_play_pcm_if_match(struct pcm *pcm)
{
    pthread_mutex_lock(&g_play_mtx);
    if (g_play_pcm == pcm) g_play_pcm = NULL;
    pthread_mutex_unlock(&g_play_mtx);
}

static void set_cap_pcm(struct pcm *pcm)
{
    pthread_mutex_lock(&g_cap_mtx);
    g_cap_pcm = pcm;
    pthread_mutex_unlock(&g_cap_mtx);
}

static struct pcm *get_cap_pcm(void)
{
    pthread_mutex_lock(&g_cap_mtx);
    struct pcm *pcm = g_cap_pcm;
    pthread_mutex_unlock(&g_cap_mtx);
    return pcm;
}

static void clear_cap_pcm_if_match(struct pcm *pcm)
{
    pthread_mutex_lock(&g_cap_mtx);
    if (g_cap_pcm == pcm) g_cap_pcm = NULL;
    pthread_mutex_unlock(&g_cap_mtx);
}

void alsa_cap_start(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit,
                    AUDIOCallback callback)
{
    (void)Volume;
    (void)bit;

    atomic_store(&gcapLoopExit, 0);

    struct pcm_config config;
    memset(&config, 0, sizeof(config));
    config.channels          = channel;
    config.rate              = 48000;
    config.period_size       = 120;
    config.period_count      = 4;
    config.format            = PCM_FORMAT_S16_LE;
    config.start_threshold   = 0;
    config.stop_threshold    = 0;
    config.silence_threshold = 0;

    unsigned int pcm_open_flags = PCM_IN | PCM_MMAP;
    struct pcm *pcm             = pcm_open(card, device, pcm_open_flags, &config);
    if (!pcm || !pcm_is_ready(pcm)) {
        fprintf(stderr, "Unable to open PCM capture device (%s)\n", pcm ? pcm_get_error(pcm) : "invalid pcm");
        if (pcm) pcm_close(pcm);
        atomic_store(&gcapLoopExit, 2);
        return;
    }

    set_cap_pcm(pcm);

    const int in_channels = channel;
    const int in_frames   = (int)pcm_get_buffer_size(pcm);
    const int in_bytes    = (int)pcm_frames_to_bytes(pcm, in_frames);

    uint8_t *buffer = (uint8_t *)malloc((size_t)in_bytes);
    if (!buffer) {
        fprintf(stderr, "Unable to allocate %d bytes\n", in_bytes);
        clear_cap_pcm_if_match(pcm);
        pcm_close(pcm);
        atomic_store(&gcapLoopExit, 2);
        return;
    }

    int16_t *ch0 = (int16_t *)malloc((size_t)in_frames * sizeof(int16_t));
    if (!ch0) {
        fprintf(stderr, "Unable to allocate ch0 buffer\n");
        free(buffer);
        clear_cap_pcm_if_match(pcm);
        pcm_close(pcm);
        atomic_store(&gcapLoopExit, 2);
        return;
    }

    SRC_STATE *src_state = NULL;
    float *in_float      = NULL;
    float *out_float     = NULL;
    int16_t *out_short   = NULL;
    int out_frames_cap   = 0;

    if (rate != 48000) {
        int err   = 0;
        src_state = src_new(SRC_SINC_FASTEST, 1, &err);
        if (!src_state) {
            fprintf(stderr, "src_new failed: %s\n", src_strerror(err));
            free(ch0);
            free(buffer);
            clear_cap_pcm_if_match(pcm);
            pcm_close(pcm);
            atomic_store(&gcapLoopExit, 2);
            return;
        }

        out_frames_cap = (int)((double)in_frames * ((double)rate / 48000.0) + 64);
        in_float       = (float *)malloc((size_t)in_frames * sizeof(float));
        out_float      = (float *)malloc((size_t)out_frames_cap * sizeof(float));
        out_short      = (int16_t *)malloc((size_t)out_frames_cap * sizeof(int16_t));

        if (!in_float || !out_float || !out_short) {
            fprintf(stderr, "Unable to allocate resample buffers\n");
            if (in_float) free(in_float);
            if (out_float) free(out_float);
            if (out_short) free(out_short);
            if (src_state) src_delete(src_state);
            free(ch0);
            free(buffer);
            clear_cap_pcm_if_match(pcm);
            pcm_close(pcm);
            atomic_store(&gcapLoopExit, 2);
            return;
        }
    }

    fprintf(stdout, "Capturing sample: in=%dch @48000Hz -> out=%dHz, S16LE\n", channel, rate);

    while (atomic_load(&gcapLoopExit) == 0) {
        int ret = pcm_readi(pcm, buffer, (unsigned int)in_frames);
        if (ret < 0) {
            if (atomic_load(&gcapLoopExit) == 1) break;
            fprintf(stderr, "Error capturing samples - %d (%s)\n", errno, strerror(errno));
            break;
        }

        const int frames_read = ret;
        if (frames_read <= 0) continue;

        const int16_t *in = (const int16_t *)buffer;
        for (int i = 0; i < frames_read; ++i) {
            ch0[i] = in[i * in_channels];
        }

        if (rate == 48000) {
            callback((const char *)ch0, frames_read * (int)sizeof(int16_t));
        } else {
            for (int i = 0; i < frames_read; ++i) {
                in_float[i] = (float)ch0[i] / 32768.0f;
            }

            SRC_DATA src_data;
            memset(&src_data, 0, sizeof(src_data));
            src_data.data_in       = in_float;
            src_data.input_frames  = frames_read;
            src_data.data_out      = out_float;
            src_data.output_frames = out_frames_cap;
            src_data.src_ratio     = (double)rate / 48000.0;
            src_data.end_of_input  = 0;

            int error = src_process(src_state, &src_data);
            if (error) {
                fprintf(stderr, "SRC error: %s\n", src_strerror(error));
                break;
            }

            int out_samples = src_data.output_frames_gen;
            if (out_samples > out_frames_cap) out_samples = out_frames_cap;

            for (int i = 0; i < out_samples; ++i) {
                float sample = out_float[i];
                if (sample > 1.0f) sample = 1.0f;
                if (sample < -1.0f) sample = -1.0f;
                out_short[i] = (int16_t)(sample * 32767.0f);
            }

            callback((const char *)out_short, out_samples * (int)sizeof(int16_t));
        }
    }

    if (src_state) src_delete(src_state);
    if (in_float) free(in_float);
    if (out_float) free(out_float);
    if (out_short) free(out_short);
    free(ch0);
    free(buffer);
    clear_cap_pcm_if_match(pcm);
    pcm_close(pcm);
    atomic_store(&gcapLoopExit, 2);
}

void alsa_close_cap(void)
{
    atomic_store(&gcapLoopExit, 1);

    struct pcm *pcm = get_cap_pcm();
    if (pcm) {
#if HAVE_PCM_STOP
        pcm_stop(pcm);
#else
        pthread_mutex_lock(&g_cap_mtx);
        if (g_cap_pcm) {
            struct pcm *to_close = g_cap_pcm;
            g_cap_pcm            = NULL;
            pthread_mutex_unlock(&g_cap_mtx);
            pcm_close(to_close);
        } else {
            pthread_mutex_unlock(&g_cap_mtx);
        }
#endif
    }
}

int alsa_cap_status(void)
{
    return atomic_load(&gcapLoopExit);
}

void alsa_play(unsigned int card, unsigned int device, float Volume, int channel, int rate, int bit, const void *data,
               int size)
{
    (void)Volume;
    (void)bit;

    atomic_store(&gplayLoopExit, 0);

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
    struct pcm *pcm             = pcm_open(card, device, pcm_open_flags, &config);
    if (!pcm || !pcm_is_ready(pcm)) {
        fprintf(stderr, "Unable to open PCM playback device (%s)\n", pcm ? pcm_get_error(pcm) : "invalid pcm");
        if (pcm) pcm_close(pcm);
        atomic_store(&gplayLoopExit, 2);
        return;
    }

    set_play_pcm(pcm);

    const uint8_t *p    = (const uint8_t *)data;
    int remaining_bytes = size;

    while (remaining_bytes > 0) {
        if (atomic_load(&gplayLoopExit) != 0) break;

        int remaining_frames = (int)pcm_bytes_to_frames(pcm, (unsigned int)remaining_bytes);

        int frames_to_write = remaining_frames;
        if (frames_to_write > (int)config.period_size) frames_to_write = (int)config.period_size;

        int written_frames = pcm_writei(pcm, p, (unsigned int)frames_to_write);

        if (written_frames < 0) {
            if (atomic_load(&gplayLoopExit) == 1) break;
            fprintf(stderr, "PCM playback error: %s\n", pcm_get_error(pcm));
            break;
        }

        if (written_frames == 0) break;

        int written_bytes = (int)pcm_frames_to_bytes(pcm, (unsigned int)written_frames);

        p += written_bytes;
        remaining_bytes -= written_bytes;
    }

    clear_play_pcm_if_match(pcm);
    pcm_close(pcm);
    atomic_store(&gplayLoopExit, 2);
}

void alsa_close_play(void)
{
    atomic_store(&gplayLoopExit, 1);

    struct pcm *pcm = get_play_pcm();
    if (pcm) {
#if HAVE_PCM_STOP
        pcm_stop(pcm);
#else
        pthread_mutex_lock(&g_play_mtx);
        if (g_play_pcm) {
            struct pcm *to_close = g_play_pcm;
            g_play_pcm           = NULL;
            pthread_mutex_unlock(&g_play_mtx);
            pcm_close(to_close);
        } else {
            pthread_mutex_unlock(&g_play_mtx);
        }
#endif
    }
}

int alsa_play_status(void)
{
    return atomic_load(&gplayLoopExit);
}