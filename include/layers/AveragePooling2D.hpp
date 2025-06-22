// include/layers/AveragePooling2D.hpp
#pragma once

#include "core/Layer.hpp"
#include <stdexcept>

class AveragePooling2D : public Layer {
private:
    int pool_h, pool_w;
    std::vector<int> input_shape;

public:
    AveragePooling2D(int pool_height, int pool_width)
        : pool_h(pool_height), pool_w(pool_width) {}

    Tensor forward(const Tensor& input) override {
        input_shape = input.shape;
        if (input.shape.size() != 4)
            throw std::invalid_argument("AveragePooling2D expects [B, C, H, W]");

        int B = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
        int out_h = H / pool_h, out_w = W / pool_w;
        Tensor output({B, C, out_h, out_w});
        int pool_size = pool_h * pool_w;

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < out_h; ++i)
                    for (int j = 0; j < out_w; ++j) {
                        float sum = 0.0f;
                        for (int m = 0; m < pool_h; ++m)
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * pool_h + m;
                                int x = j * pool_w + n;
                                int idx = ((b * C + c) * H + y) * W + x;
                                sum += input.data[idx];
                            }
                        output.at({b, c, i, j}) = sum / pool_size;
                    }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape);
        grad_input.fill(0.0f);

        int B = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
        int out_h = H / pool_h, out_w = W / pool_w;
        float scale = 1.0f / (pool_h * pool_w);

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < out_h; ++i)
                    for (int j = 0; j < out_w; ++j) {
                        float grad = grad_output.at({b, c, i, j}) * scale;
                        for (int m = 0; m < pool_h; ++m)
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * pool_h + m;
                                int x = j * pool_w + n;
                                int idx = ((b * C + c) * H + y) * W + x;
                                grad_input.data[idx] += grad;
                            }
                    }

        return grad_input;
    }

    void update_weights(Optimizer*) override {}
};
