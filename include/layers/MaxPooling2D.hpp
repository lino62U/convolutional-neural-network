#pragma once

#include "core/Layer.hpp"
#include <limits>
#include <stdexcept>
#include <cmath>


// MaxPooling layer
class MaxPooling2D : public Layer {
private:
    int pool_size;
    int stride;
    Tensor input_cache;
    std::vector<std::vector<int>> max_indices; // Store indices of max values for backward pass

public:
    MaxPooling2D(int size, int str) : pool_size(size), stride(str) {
        if (size <= 0 || str <= 0) {
            throw std::runtime_error("Invalid max pooling parameters");
        }
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 4) {
            throw std::runtime_error("Invalid input shape for max pooling");
        }

        int batch_size = input.shape[0];
        int channels = input.shape[1];
        int in_height = input.shape[2];
        int in_width = input.shape[3];

        // Compute output dimensions
        int out_height = (in_height - pool_size) / stride + 1;
        int out_width = (in_width - pool_size) / stride + 1;
        if (out_height <= 0 || out_width <= 0) {
            throw std::runtime_error("Invalid output dimensions for max pooling");
        }

        input_cache = input;
        max_indices.clear();
        max_indices.resize(batch_size * channels * out_height * out_width);

        std::vector<float> output_data(batch_size * channels * out_height * out_width);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_idx = 0;
                        for (int ph = 0; ph < pool_size; ++ph) {
                            for (int pw = 0; pw < pool_size; ++pw) {
                                int input_h = h * stride + ph;
                                int input_w = w * stride + pw;
                                if (input_h < in_height && input_w < in_width) {
                                    float val = input.data[n * channels * in_height * in_width +
                                        c * in_height * in_width + input_h * in_width + input_w];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = ph * pool_size + pw;
                                    }
                                }
                            }
                        }
                        int output_idx = n * channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
                        output_data[output_idx] = max_val;
                        max_indices[output_idx] = std::vector<int>{n, c, h * stride + max_idx / pool_size, w * stride + max_idx % pool_size};
                    }
                }
            }
        }

        return Tensor(output_data, {batch_size, channels, out_height, out_width});
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape[2] != (input_cache.shape[2] - pool_size) / stride + 1 ||
            grad_output.shape[3] != (input_cache.shape[3] - pool_size) / stride + 1) {
            throw std::runtime_error("Gradient shape mismatch in max pooling backward");
        }

        int batch_size = input_cache.shape[0];
        int channels = input_cache.shape[1];
        int in_height = input_cache.shape[2];
        int in_width = input_cache.shape[3];
        int out_height = grad_output.shape[2];
        int out_width = grad_output.shape[3];

        std::vector<float> grad_input_data(batch_size * channels * in_height * in_width, 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        int output_idx = n * channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
                        const auto& max_pos = max_indices[output_idx];
                        int input_h = max_pos[2];
                        int input_w = max_pos[3];
                        grad_input_data[max_pos[0] * channels * in_height * in_width +
                                       max_pos[1] * in_height * in_width + input_h * in_width + input_w] +=
                            grad_output.data[output_idx];
                    }
                }
            }
        }

        return Tensor(grad_input_data, input_cache.shape);
    }

    void update_weights(Optimizer* optimizer) override {
        // No trainable parameters
    }

    size_t num_params() const override {
        return 0;
    }
};
