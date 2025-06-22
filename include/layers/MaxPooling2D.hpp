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
        input_shape = input.shape; // [channels, height, width]
        if (input.shape.size() != 3) {
            throw std::invalid_argument("MaxPooling2D expects input with shape [C, H, W]");
        }

        int channels = input.shape[0];
        int height = input.shape[1];
        int width = input.shape[2];

        int out_h = height / pool_h;
        int out_w = width / pool_w;

        Tensor output({channels, out_h, out_w});
        mask = Tensor(input.shape);

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    float max_val = std::numeric_limits<float>::lowest();
                    int max_idx = -1;

                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            int y = i * pool_h + m;
                            int x = j * pool_w + n;
                            int idx = c * height * width + y * width + x;
                            float val = input.data[idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = idx;
                            }
                        }
                    }

                    output.data[c * out_h * out_w + i * out_w + j] = max_val;
                    if (max_idx >= 0) {
                        mask.data[max_idx] = 1.0f;
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape);
        std::fill(grad_input.data.begin(), grad_input.data.end(), 0.0f);

        int channels = input_shape[0];
        int height = input_shape[1];
        int width = input_shape[2];
        int out_h = height / pool_h;
        int out_w = width / pool_w;

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    int out_idx = c * out_h * out_w + i * out_w + j;
                    float grad = grad_output.data[out_idx];

                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            int y = i * pool_h + m;
                            int x = j * pool_w + n;
                            int idx = c * height * width + y * width + x;
                            if (mask.data[idx] == 1.0f) {
                                grad_input.data[idx] = grad;
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {}
};
