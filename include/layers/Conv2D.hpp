#pragma once

#include "core/Layer.hpp"
#include "core/Tensor.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>
#include <algorithm>
#include <string>

class Conv2D : public Layer {
private:
    Tensor filters; // Shape: {num_filters, in_channels, kernel_height, kernel_width}
    Tensor bias; // Shape: {num_filters}
    int padding;
    int stride;
    std::shared_ptr<Activation> activation;
    Tensor input_cache;
    Tensor z_cache; // Pre-activation values
    Tensor activation_cache; // Post-activation values
    Tensor filters_grad;
    Tensor bias_grad;
    std::mt19937 rng;

    // Initialize filters using He initialization
    void initialize_filters(int in_channels, int num_filters, int kernel_size) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / (in_channels * kernel_size * kernel_size)));
        std::vector<float> f_data(num_filters * in_channels * kernel_size * kernel_size);
        for (float& f : f_data) {
            f = dist(rng);
        }
        filters = Tensor(f_data, {num_filters, in_channels, kernel_size, kernel_size});
        
        std::vector<float> b_data(num_filters, 0.0f);
        bias = Tensor(b_data, {num_filters});
    }

public:
    Conv2D(int in_channels, int num_filters, int kernel_size, int pad, int str, std::shared_ptr<Activation> act)
        : padding(pad), stride(str), activation(act), rng(std::random_device{}()) {
        if (pad < 0 || str <= 0 || kernel_size <= 0) {
            throw std::runtime_error("Invalid convolution parameters");
        }
        initialize_filters(in_channels, num_filters, kernel_size);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 4 || input.shape[1] != filters.shape[1]) {
            throw std::runtime_error("Invalid input shape for convolution");
        }

        int batch_size = input.shape[0];
        int in_channels = input.shape[1];
        int in_height = input.shape[2];
        int in_width = input.shape[3];
        int num_filters = filters.shape[0];
        int kernel_height = filters.shape[2];
        int kernel_width = filters.shape[3];

        // Compute output dimensions
        int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        if (out_height <= 0 || out_width <= 0) {
            throw std::runtime_error("Invalid output dimensions");
        }

        // Pad input
        std::vector<float> padded_input_data(batch_size * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding), 0.0f);
        Tensor padded_input(padded_input_data, {batch_size, in_channels, in_height + 2 * padding, in_width + 2 * padding});
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        padded_input.data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                         c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                         (h + padding) * (in_width + 2 * padding) + (w + padding)] =
                            input.data[n * in_channels * in_height * in_width + c * in_height * in_width + h * in_width + w];
                    }
                }
            }
        }

        input_cache = padded_input; // Store padded input for backward pass

        // Perform convolution
        std::vector<float> output_data(batch_size * num_filters * out_height * out_width);
        for (int n = 0; n < batch_size; ++n) {
            for (int f = 0; f < num_filters; ++f) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        float sum = bias.data[f];
                        for (int c = 0; c < in_channels; ++c) {
                            for (int kh = 0; kh < kernel_height; ++kh) {
                                for (int kw = 0; kw < kernel_width; ++kw) {
                                    int input_h = h * stride + kh;
                                    int input_w = w * stride + kw;
                                    sum += padded_input.data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                            c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                            input_h * (in_width + 2 * padding) + input_w] *
                                           filters.data[f * in_channels * kernel_height * kernel_width +
                                                        c * kernel_height * kernel_width + kh * kernel_width + kw];
                                }
                            }
                        }
                        output_data[n * num_filters * out_height * out_width + f * out_height * out_width + h * out_width + w] = sum;
                    }
                }
            }
        }

        z_cache = Tensor(output_data, {batch_size, num_filters, out_height, out_width});
        activation_cache = activation->apply_batch(z_cache);
        return activation_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape != activation_cache.shape) {
            throw std::runtime_error("Gradient shape mismatch in convolution backward");
        }

        int batch_size = input_cache.shape[0];
        int in_channels = input_cache.shape[1];
        int in_height = input_cache.shape[2] - 2 * padding;
        int in_width = input_cache.shape[3] - 2 * padding;
        int num_filters = filters.shape[0];
        int kernel_height = filters.shape[2];
        int kernel_width = filters.shape[3];
        int out_height = grad_output.shape[2];
        int out_width = grad_output.shape[3];

        // Compute gradient w.r.t. pre-activation values
        Tensor grad_z = activation->derivative_batch(z_cache, activation_cache);
        for (size_t i = 0; i < grad_z.data.size(); ++i) {
            grad_z.data[i] *= grad_output.data[i];
        }

        // Compute gradients for filters
        std::vector<float> grad_filters_data(filters.size(), 0.0f);
        for (int f = 0; f < num_filters; ++f) {
            for (int c = 0; c < in_channels; ++c) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        float sum = 0.0f;
                        for (int n = 0; n < batch_size; ++n) {
                            for (int h = 0; h < out_height; ++h) {
                                for (int w = 0; w < out_width; ++w) {
                                    sum += input_cache.data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                           c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                           (h * stride + kh) * (in_width + 2 * padding) + (w * stride + kw)] *
                                           grad_z.data[n * num_filters * out_height * out_width +
                                                       f * out_height * out_width + h * out_width + w];
                                }
                            }
                        }
                        grad_filters_data[f * in_channels * kernel_height * kernel_width +
                                         c * kernel_height * kernel_width + kh * kernel_width + kw] = sum;
                    }
                }
            }
        }
        filters_grad = Tensor(grad_filters_data, filters.shape);

        // Compute gradients for bias
        std::vector<float> grad_bias_data(num_filters, 0.0f);
        for (int f = 0; f < num_filters; ++f) {
            for (int n = 0; n < batch_size; ++n) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        grad_bias_data[f] += grad_z.data[n * num_filters * out_height * out_width +
                                                        f * out_height * out_width + h * out_width + w];
                    }
                }
            }
        }
        bias_grad = Tensor(grad_bias_data, bias.shape);

        // Compute gradient w.r.t. input
        std::vector<float> grad_input_data(input_cache.size(), 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        for (int f = 0; f < num_filters; ++f) {
                            for (int kh = 0; kh < kernel_height; ++kh) {
                                for (int kw = 0; kw < kernel_width; ++kw) {
                                    int out_h = (h - kh + padding) / stride;
                                    int out_w = (w - kw + padding) / stride;
                                    if (out_h >= 0 && out_h < out_height && (h - kh + padding) % stride == 0 &&
                                        out_w >= 0 && out_w < out_width && (w - kw + padding) % stride == 0) {
                                        grad_input_data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                       c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                       (h + padding) * (in_width + 2 * padding) + (w + padding)] +=
                                            grad_z.data[n * num_filters * out_height * out_width +
                                                       f * out_height * out_width + out_h * out_width + out_w] *
                                            filters.data[f * in_channels * kernel_height * kernel_width +
                                                        c * kernel_height * kernel_width + kh * kernel_width + kw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extract unpadded gradient
        std::vector<float> grad_input_unpadded(batch_size * in_channels * in_height * in_width);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        grad_input_unpadded[n * in_channels * in_height * in_width + c * in_height * in_width + h * in_width + w] =
                            grad_input_data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                           c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                           (h + padding) * (in_width + 2 * padding) + (w + padding)];
                    }
                }
            }
        }

        return Tensor(grad_input_unpadded, {batch_size, in_channels, in_height, in_width});
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(filters, filters_grad);
        optimizer->update(bias, bias_grad);
    }

    size_t num_params() const override {
        return filters.total_elements() + bias.total_elements();
    }
};