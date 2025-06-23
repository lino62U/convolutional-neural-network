// include/layers/MaxPooling2D.hpp
#pragma once

#include "core/Layer.hpp"
#include <limits>
#include <stdexcept>

class MaxPooling2D : public Layer {
private:
    int pool_h, pool_w;
    std::vector<int> input_shape;
    Tensor mask;

public:
    MaxPooling2D(int pool_height, int pool_width)
        : pool_h(pool_height), pool_w(pool_width) {}

    Tensor forward(const Tensor& input) override {
        input_shape = input.shape; // [B, C, H, W]
        if (input.shape.size() != 4)
            throw std::invalid_argument("MaxPooling2D expects input with shape [B, C, H, W]");

        int batch = input.shape[0];
        int channels = input.shape[1];
        int height = input.shape[2];
        int width = input.shape[3];

        int out_h = height / pool_h;
        int out_w = width / pool_w;

        Tensor output({batch, channels, out_h, out_w});
        mask = Tensor(input.shape);
        mask.fill(0.0f);

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        float max_val = std::numeric_limits<float>::lowest();
                        int max_idx = -1;

                        for (int m = 0; m < pool_h; ++m) {
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * pool_h + m;
                                int x = j * pool_w + n;
                                int idx = ((b * channels + c) * height + y) * width + x;
                                float val = input.data[idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = idx;
                                }
                            }
                        }

                        int out_idx = ((b * channels + c) * out_h + i) * out_w + j;
                        output.data[out_idx] = max_val;
                        if (max_idx >= 0) {
                            mask.data[max_idx] = 1.0f;
                        }
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape);
        grad_input.fill(0.0f);

        int batch = input_shape[0];
        int channels = input_shape[1];
        int height = input_shape[2];
        int width = input_shape[3];
        int out_h = height / pool_h;
        int out_w = width / pool_w;

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        int out_idx = ((b * channels + c) * out_h + i) * out_w + j;
                        float grad = grad_output.data[out_idx];

                        for (int m = 0; m < pool_h; ++m) {
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * pool_h + m;
                                int x = j * pool_w + n;
                                int idx = ((b * channels + c) * height + y) * width + x;
                                if (mask.data[idx] == 1.0f) {
                                    grad_input.data[idx] = grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    size_t num_params() const override {
        return 0;  // Estas capas no tienen parámetros entrenables
    }
    
    void update_weights(Optimizer* optimizer) override {}
};
