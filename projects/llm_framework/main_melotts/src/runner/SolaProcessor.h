#ifndef SOLA_PROCESSOR_H
#define SOLA_PROCESSOR_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

/**
 * SolaProcessor - Synchronous Overlap-Add method for audio frame processing
 *
 * This class provides functionality for smoothly concatenating audio frames
 * using the SOLA algorithm, which finds optimal alignment points between
 * consecutive frames and applies crossfading for smooth transitions.
 */
class SolaProcessor {
public:
    /**
     * Constructor
     *
     * @param padFrames Number of padding frames at the beginning and end
     * @param samplesPerFrame Number of audio samples in each frame
     */
    SolaProcessor(int padFrames, int samplesPerFrame)
        : pad_frames_(padFrames), samples_per_frame_(samplesPerFrame), first_frame_(true)
    {
        Initialize();
    }

    /**
     * Reset the processor to its initial state
     */
    void Reset()
    {
        first_frame_ = true;
        std::fill(sola_buffer_.begin(), sola_buffer_.end(), 0.0f);
    }

    /**
     * Process a single audio frame
     *
     * @param decoder_output Raw audio data from decoder
     * @param frameIndex Current frame index
     * @param totalFrames Total number of frames
     * @param actualFrameLen Actual length of the frame
     * @return Processed audio samples
     */
    std::vector<float> ProcessFrame(const std::vector<float>& decoder_output, int frameIndex, int totalFrames,
                                    int actualFrameLen)
    {
        std::vector<float> processed_output;

        if (first_frame_) {
            // Special handling for the first frame
            ProcessFirstFrame(decoder_output, processed_output, actualFrameLen);
            first_frame_ = false;
        } else {
            // Process subsequent frames with SOLA algorithm
            ProcessSubsequentFrame(decoder_output, processed_output, frameIndex, totalFrames, actualFrameLen);
        }

        return processed_output;
    }

private:
    /**
     * Initialize the SOLA processor parameters and buffers
     */
    void Initialize()
    {
        // Calculate SOLA parameters
        sola_buffer_frame_ = pad_frames_ * samples_per_frame_;
        sola_search_frame_ = pad_frames_ * samples_per_frame_;
        effective_frames_  = 0;  // Will be set during frame processing

        // Create fade-in and fade-out windows
        fade_in_window_.resize(sola_buffer_frame_);
        fade_out_window_.resize(sola_buffer_frame_);

        for (int i = 0; i < sola_buffer_frame_; i++) {
            fade_in_window_[i]  = static_cast<float>(i) / sola_buffer_frame_;
            fade_out_window_[i] = 1.0f - fade_in_window_[i];
        }

        // Initialize SOLA buffer
        sola_buffer_.resize(sola_buffer_frame_, 0.0f);
    }

    /**
     * Process the first audio frame
     *
     * @param decoder_output Raw audio data from decoder
     * @param processed_output Output buffer for processed audio
     * @param actualFrameLen Actual length of the frame
     */
    void ProcessFirstFrame(const std::vector<float>& decoder_output, std::vector<float>& processed_output,
                           int actualFrameLen)
    {
        int audio_start = pad_frames_ * samples_per_frame_;
        int audio_len   = (actualFrameLen - 2 * pad_frames_) * samples_per_frame_;

        // Boundary check
        audio_len = std::min(audio_len, static_cast<int>(decoder_output.size() - audio_start));

        // Add first frame data to output
        processed_output.insert(processed_output.end(), decoder_output.begin() + audio_start,
                                decoder_output.begin() + audio_start + audio_len);

        // Save the end part to SOLA buffer for next frame alignment
        int buffer_start = audio_start + audio_len;
        if (buffer_start + sola_buffer_frame_ <= decoder_output.size()) {
            std::copy(decoder_output.begin() + buffer_start, decoder_output.begin() + buffer_start + sola_buffer_frame_,
                      sola_buffer_.begin());
        }
    }

    /**
     * Process subsequent audio frames using SOLA algorithm
     *
     * @param decoder_output Raw audio data from decoder
     * @param processed_output Output buffer for processed audio
     * @param frameIndex Current frame index
     * @param totalFrames Total number of frames
     * @param actualFrameLen Actual length of the frame
     */
    void ProcessSubsequentFrame(const std::vector<float>& decoder_output, std::vector<float>& processed_output,
                                int frameIndex, int totalFrames, int actualFrameLen)
    {
        int audio_start = pad_frames_ * samples_per_frame_;

        // 1. Prepare search window
        std::vector<float> search_window(sola_buffer_frame_ + sola_search_frame_);
        std::copy(decoder_output.begin() + audio_start, decoder_output.begin() + audio_start + search_window.size(),
                  search_window.begin());

        // 2. Find best alignment point (compute cross-correlation)
        int best_offset = FindBestOffset(search_window);

        // 3. Apply alignment offset
        int aligned_start = audio_start + best_offset;

        // 4. Create smooth transition
        std::vector<float> crossfade_region = CreateCrossfade(decoder_output, aligned_start);

        // 5. Add crossfade region to output
        processed_output.insert(processed_output.end(), crossfade_region.begin(), crossfade_region.end());

        // 6. Add remaining valid audio data
        AddRemainingAudio(decoder_output, processed_output, aligned_start, frameIndex, totalFrames, actualFrameLen);
    }

    /**
     * Find the best alignment offset using normalized cross-correlation
     *
     * @param search_window Window of audio samples to search in
     * @return Optimal offset for alignment
     */
    int FindBestOffset(const std::vector<float>& search_window)
    {
        int best_offset        = 0;
        float best_correlation = -1.0f;

        for (int offset = 0; offset <= sola_search_frame_; offset++) {
            float correlation = 0.0f;
            float energy      = 0.0f;

            for (int j = 0; j < sola_buffer_frame_; j++) {
                correlation += sola_buffer_[j] * search_window[j + offset];
                energy += search_window[j + offset] * search_window[j + offset];
            }

            // Normalize correlation
            float normalized_correlation = (energy > 1e-8) ? correlation / std::sqrt(energy) : 0.0f;

            if (normalized_correlation > best_correlation) {
                best_correlation = normalized_correlation;
                best_offset      = offset;
            }
        }

        return best_offset;
    }

    /**
     * Create crossfade transition region
     *
     * @param decoder_output Raw audio data from decoder
     * @param aligned_start Starting point after alignment
     * @return Crossfaded audio samples
     */
    std::vector<float> CreateCrossfade(const std::vector<float>& decoder_output, int aligned_start)
    {
        std::vector<float> crossfade_region(sola_buffer_frame_);

        for (int j = 0; j < sola_buffer_frame_; j++) {
            // Apply fade-in and fade-out window functions
            crossfade_region[j] =
                decoder_output[aligned_start + j] * fade_in_window_[j] + sola_buffer_[j] * fade_out_window_[j];
        }

        return crossfade_region;
    }

    /**
     * Add remaining audio data and update buffer
     *
     * @param decoder_output Raw audio data from decoder
     * @param processed_output Output buffer for processed audio
     * @param aligned_start Starting point after alignment
     * @param frameIndex Current frame index
     * @param totalFrames Total number of frames
     * @param actualFrameLen Actual length of the frame
     */
    void AddRemainingAudio(const std::vector<float>& decoder_output, std::vector<float>& processed_output,
                           int aligned_start, int frameIndex, int totalFrames, int actualFrameLen)
    {
        int remaining_start = aligned_start + sola_buffer_frame_;
        int remaining_len   = (actualFrameLen - 2 * pad_frames_) * samples_per_frame_ - sola_buffer_frame_;

        // Boundary check
        remaining_len = std::min(remaining_len, static_cast<int>(decoder_output.size() - remaining_start));

        if (remaining_len > 0) {
            processed_output.insert(processed_output.end(), decoder_output.begin() + remaining_start,
                                    decoder_output.begin() + remaining_start + remaining_len);
        }

        // Update SOLA buffer
        UpdateSolaBuffer(decoder_output, remaining_start + remaining_len);
    }

    /**
     * Update SOLA buffer with new audio data
     *
     * @param decoder_output Raw audio data from decoder
     * @param buffer_start Starting point for the new buffer data
     */
    void UpdateSolaBuffer(const std::vector<float>& decoder_output, int buffer_start)
    {
        // Check if there's enough data for the next buffer
        if (buffer_start + sola_buffer_frame_ <= decoder_output.size()) {
            std::copy(decoder_output.begin() + buffer_start, decoder_output.begin() + buffer_start + sola_buffer_frame_,
                      sola_buffer_.begin());
        } else {
            // Fill with zeros if not enough data
            int avail = static_cast<int>(decoder_output.size() - buffer_start);
            if (avail > 0) {
                std::copy(decoder_output.begin() + buffer_start, decoder_output.end(), sola_buffer_.begin());
            }
            std::fill(sola_buffer_.begin() + avail, sola_buffer_.end(), 0.0f);
        }
    }

private:
    int pad_frames_;         // Number of padding frames
    int samples_per_frame_;  // Number of samples per frame
    int effective_frames_;   // Number of effective frames
    int sola_buffer_frame_;  // SOLA buffer length
    int sola_search_frame_;  // SOLA search window length

    std::vector<float> fade_in_window_;   // Fade-in window
    std::vector<float> fade_out_window_;  // Fade-out window
    std::vector<float> sola_buffer_;      // SOLA buffer

    bool first_frame_;  // Flag for first frame processing
};

#endif  // SOLA_PROCESSOR_H
