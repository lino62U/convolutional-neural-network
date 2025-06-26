#pragma once

#include "core/Layer.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

class AveragePooling2D : public Layer {
private:
    int pool_h, pool_w;
    int stride;
    std::string padding_type;
    int pad_h = 0, pad_w = 0;
    std::vector<int> input_shape;

public:
    AveragePooling2D(int pool_height, int pool_width, int stride_ = 1, const std::string& padding = "valid")
        : pool_h(pool_height), pool_w(pool_width),
          stride(stride_), padding_type(padding) {}

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 4) {
            throw std::runtime_error("AveragePooling2D solo soporta tensores 4D");
        }

        int batch = input.shape[0];
        int channels = input.shape[1];
        int height = input.shape[2];
        int width = input.shape[3];

        int out_h = (height - pool_h) / stride + 1;
        int out_w = (width - pool_w) / stride + 1;

        if (out_h <= 0 || out_w <= 0) {
            throw std::runtime_error("Dimensiones de salida inválidas en AveragePooling2D");
        }

        std::vector<float> output_data(batch * channels * out_h * out_w, 0.0f);

        for (int n = 0; n < batch; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        float sum = 0.0f;
                        for (int ki = 0; ki < pool_h; ++ki) {
                            for (int kj = 0; kj < pool_w; ++kj) {
                                int in_i = i * stride + ki;
                                int in_j = j * stride + kj;
                                if (in_i < height && in_j < width) {
                                    int idx = n * channels * height * width +
                                            c * height * width +
                                            in_i * width + in_j;
                                    sum += input.data[idx];
                                }
                            }
                        }
                        float avg = sum / (pool_h * pool_w);
                        int out_idx = n * channels * out_h * out_w +
                                    c * out_h * out_w +
                                    i * out_w + j;
                        output_data[out_idx] = avg;
                    }
                }
            }
        }

        return Tensor(output_data, {batch, channels, out_h, out_w});
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape.size() != 4) {
            throw std::runtime_error("grad_output debe tener 4 dimensiones");
        }

        int batch = input_shape[0];
        int channels = input_shape[1];
        int height = input_shape[2];
        int width = input_shape[3];

        int out_h = grad_output.shape[2];
        int out_w = grad_output.shape[3];

        std::vector<float> grad_input_data(batch * channels * height * width, 0.0f);

        for (int n = 0; n < batch; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        float grad = grad_output.data[n * channels * out_h * out_w +
                                                    c * out_h * out_w +
                                                    i * out_w + j];
                        float avg_grad = grad / (pool_h * pool_w);

                        for (int ki = 0; ki < pool_h; ++ki) {
                            for (int kj = 0; kj < pool_w; ++kj) {
                                int in_i = i * stride + ki;
                                int in_j = j * stride + kj;
                                if (in_i < height && in_j < width) {
                                    int idx = n * channels * height * width +
                                            c * height * width +
                                            in_i * width + in_j;
                                    grad_input_data[idx] += avg_grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        return Tensor(grad_input_data, input_shape);
    }



    size_t num_params() const override { return 0; }

    void update_weights(Optimizer*) override {}
};
