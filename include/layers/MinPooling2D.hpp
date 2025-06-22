// include/layers/MinPooling2D.hpp
#pragma once

#include "core/Layer.hpp"
#include <limits>
#include <stdexcept>

class MinPooling2D : public Layer {
private:
    int pool_h, pool_w;
    std::vector<int> input_shape;
    Tensor mask;

public:
    MinPooling2D(int pool_height, int pool_width)
        : pool_h(pool_height), pool_w(pool_width) {}

    Tensor forward(const Tensor& input) override {
        input_shape = input.shape;
        if (input.shape.size() != 4)
            throw std::invalid_argument("MinPooling2D expects [B, C, H, W]");

        int B = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
        int out_h = H / pool_h, out_w = W / pool_w;

        Tensor output({B, C, out_h, out_w});
        mask = Tensor(input.shape);
        mask.fill(0.0f);

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < out_h; ++i)
                    for (int j = 0; j < out_w; ++j) {
                        float min_val = std::numeric_limits<float>::max();
                        int min_idx = -1;
                        for (int m = 0; m < pool_h; ++m)
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * pool_h + m;
                                int x = j * pool_w + n;
                                int idx = ((b * C + c) * H + y) * W + x;
                                float val = input.data[idx];
                                if (val < min_val) {
                                    min_val = val;
                                    min_idx = idx;
                                }
                            }
                        output.at({b, c, i, j}) = min_val;
                        if (min_idx >= 0) mask.data[min_idx] = 1.0f;
                    }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape);
        grad_input.fill(0.0f);

        int B = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
        int out_h = H / pool_h, out_w = W / pool_w;

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < out_h; ++i)
                    for (int j = 0; j < out_w; ++j) {
                        float grad = grad_output.at({b, c, i, j});
                        for (int m = 0; m < pool_h; ++m)
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * pool_h + m;
                                int x = j * pool_w + n;
                                int idx = ((b * C + c) * H + y) * W + x;
                                if (mask.data[idx] == 1.0f)
                                    grad_input.data[idx] = grad;
                            }
                    }

        return grad_input;
    }

    void update_weights(Optimizer*) override {}
};
