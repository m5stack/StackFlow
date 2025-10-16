// sampling.h
#ifndef SAMPLING_H
#define SAMPLING_H

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream> // For debugging, can be removed
#include <stdexcept>

namespace sampling {
    
    // Numerically stable softmax implementation [[32]]
    std::vector<float> softmax_stable(const std::vector<float>& logits) {
        if (logits.empty()) return {};

        // Find the maximum value for numerical stability [[32]]
        float max_val = *std::max_element(logits.begin(), logits.end());

        std::vector<float> exp_values(logits.size());
        float sum_exp = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            exp_values[i] = std::exp(logits[i] - max_val); // Subtract max for stability [[32]]
            sum_exp += exp_values[i];
        }

        // Normalize
        if (sum_exp > 0.0f) {
            for (float& val : exp_values) {
                val /= sum_exp;
            }
        } else {
            // Handle case where all logits are very negative (sum_exp ~ 0)
            // Assign uniform probability
            float uniform_prob = 1.0f / static_cast<float>(exp_values.size());
            for (float& val : exp_values) {
                val = uniform_prob;
            }
        }
        return exp_values;
    }

    // Sort indices based on values in descending order [[29]]
    std::vector<size_t> sort_indices_desc(const std::vector<float>& v) {
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0); // Fill idx with 0, 1, ..., v.size()-1 [[23]]

        // Sort indices based on the corresponding values in 'v' in descending order [[24]]
        std::stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) {return v[i1] > v[i2];}); // [[29]]

        return idx;
    }

    // Multinomial sampling with replacement [[45]]
    int sample_multinomial(const std::vector<float>& probabilities, std::mt19937& gen) {
        if (probabilities.empty()) {
            throw std::invalid_argument("Cannot sample from an empty probability distribution.");
        }

        std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end()); // [[13]]
        return dist(gen);
    }

    // --- Core Sampling Functions ---

    // Nucleus (Top-p) Sampling with Top-k filtering
    int nucleus_sampling(const std::vector<float>& weighted_scores, float top_p = 0.8f, int top_k = 25) {
        if (weighted_scores.empty()) {
            throw std::invalid_argument("weighted_scores cannot be empty.");
        }

        // 1. Apply softmax to get probabilities
        std::vector<float> probs = softmax_stable(weighted_scores);

        // 2. Get sorted indices (descending) [[29]]
        std::vector<size_t> sorted_indices = sort_indices_desc(probs);

        // 3. Apply Top-p and Top-k filtering
        std::vector<float> filtered_probs;
        std::vector<size_t> filtered_indices;
        float cum_prob = 0.0f;

        int actual_top_k = std::min(top_k, static_cast<int>(sorted_indices.size()));

        for (int i = 0; i < actual_top_k; ++i) {
            size_t idx = sorted_indices[i];
            float prob = probs[idx];
            if (cum_prob < top_p && static_cast<int>(filtered_probs.size()) < top_k) {
                cum_prob += prob;
                filtered_probs.push_back(prob);
                filtered_indices.push_back(idx);
            } else {
                break; // Stop if cumulative probability exceeds top_p or reached top_k limit
            }
        }

        if (filtered_probs.empty()) {
            // This can happen if the first element's probability is >= top_p
            // or if top_k is 0. Fall back to sampling from the top element.
            // Or if all probabilities were 0 (handled by softmax_stable).
            filtered_probs.push_back(1.0f);
            filtered_indices.push_back(sorted_indices[0]);
        }

        // 4. Re-normalize the filtered probabilities (like PyTorch does)
        // This is crucial to match Python behavior
        float sum_filtered = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        if (sum_filtered > 0.0f) {
            for (float& prob : filtered_probs) {
                prob /= sum_filtered;
            }
        }

        // 5. Sample from the filtered distribution
        static std::random_device rd; // Static to seed once
        static std::mt19937 gen(rd());
        int sampled_index_in_filtered = sample_multinomial(filtered_probs, gen);

        // 6. Return the original index
        return static_cast<int>(filtered_indices[sampled_index_in_filtered]);
    }

    // Random Sampling (equivalent to Top-1 with temperature -> 1)
    int random_sampling(const std::vector<float>& weighted_scores) {
        if (weighted_scores.empty()) {
            throw std::invalid_argument("weighted_scores cannot be empty.");
        }
        std::vector<float> probs = softmax_stable(weighted_scores);

        static std::random_device rd;
        static std::mt19937 gen(rd());
        return sample_multinomial(probs, gen);
    }

    // Repetition-Aware Sampling (RAS)
    int ras_sampling(const std::vector<float>& weighted_scores,
                    const std::vector<int>& decoded_tokens,
                    int speech_token_size, // Assuming this is passed or part of context
                    float top_p = 0.8f, int top_k = 25,
                    int win_size = 10, float tau_r = 0.1f) {

        // 1. Perform Nucleus Sampling
        int top_id = nucleus_sampling(weighted_scores, top_p, top_k);

        // 2. Check for repetition
        int rep_num = 0;
        int window_start = std::max(0, static_cast<int>(decoded_tokens.size()) - win_size);
        for (size_t i = window_start; i < decoded_tokens.size(); ++i) {
            if (decoded_tokens[i] == top_id) {
                rep_num++;
            }
        }

        // 3. If repetition threshold is met, fallback to random sampling
        if (rep_num >= static_cast<int>(win_size * tau_r)) {
            top_id = random_sampling(weighted_scores);
        }

        return top_id;
    }

    // Main sampling function with EOS handling
    int sampling_ids(const std::vector<float>& weighted_scores,
                    const std::vector<int>& decoded_tokens,
                    int speech_token_size, // Assuming this is passed or part of context
                    bool ignore_eos = true,
                    int max_trials = 100) {

        static std::random_device rd;
        static std::mt19937 gen(rd()); // Static for efficiency in loops

        int num_trials = 0;
        int top_id = -1; // Initialize

        while (true) {
            top_id = ras_sampling(weighted_scores, decoded_tokens, speech_token_size);

            // Check EOS condition
            if (!ignore_eos || (speech_token_size < 0 || top_id != speech_token_size)) {
                break; // Accept the sample if EOS is not ignored, or if it's not the EOS token
            }

            num_trials++;
            if (num_trials > max_trials) {
                throw std::runtime_error("sampling reaches max_trials " + std::to_string(max_trials) +
                                        " and still gets eos when ignore_eos is True, check your input!");
            }
        }
        return top_id;
    }



} // namespace sampling

#endif // SAMPLING_H