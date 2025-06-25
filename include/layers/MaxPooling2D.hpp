#pragma once

#include "core/Layer.hpp"
#include <limits>
#include <stdexcept>
#include <cmath>

class MaxPooling2D : public Layer {
private:
    int pool_h, pool_w;
    int stride;
    std::string padding_type;
    int pad_h, pad_w;
    std::vector<int> input_shape;
    Tensor mask;

public:
    MaxPooling2D(int pool_height, int pool_width, int stride_ = 1, const std::string& padding = "valid")
        : pool_h(pool_height), pool_w(pool_width),
          stride(stride_), padding_type(padding),
          pad_h(0), pad_w(0) {}

    Tensor forward(const Tensor& input) override {
        input_shape = input.shape; // [B, C, H, W]
        if (input.shape.size() != 4)
            throw std::invalid_argument("MaxPooling2D expects input shape [B, C, H, W]");

        int B = input.shape[0], C = input.shape[1];
        int H = input.shape[2], W = input.shape[3];

        // Calcular padding
        if (padding_type == "same") {
            int out_h = static_cast<int>(std::ceil(float(H) / stride));
            int out_w = static_cast<int>(std::ceil(float(W) / stride));
            int pad_total_h = std::max(0, (out_h - 1) * stride + pool_h - H);
            int pad_total_w = std::max(0, (out_w - 1) * stride + pool_w - W);
            pad_h = pad_total_h / 2;
            pad_w = pad_total_w / 2;
        } else if (padding_type == "valid") {
            pad_h = 0;
            pad_w = 0;
        } else {
            throw std::invalid_argument("Unknown padding type: " + padding_type);
        }

        int H_pad = H + 2 * pad_h;
        int W_pad = W + 2 * pad_w;

        int out_h = (H_pad - pool_h) / stride + 1;
        int out_w = (W_pad - pool_w) / stride + 1;

        Tensor output({B, C, out_h, out_w});
        Tensor padded({B, C, H_pad, W_pad});
        padded.fill(0.0f);
        mask = Tensor({B, C, H_pad, W_pad});
        mask.fill(0.0f);

        // Copiar input con padding
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        padded.at({b, c, h + pad_h, w + pad_w}) = input.at({b, c, h, w});

        // Max pooling
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        float max_val = std::numeric_limits<float>::lowest();
                        int max_y = -1, max_x = -1;

                        for (int m = 0; m < pool_h; ++m) {
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * stride + m;
                                int x = j * stride + n;
                                float val = padded.at({b, c, y, x});
                                if (val > max_val) {
                                    max_val = val;
                                    max_y = y;
                                    max_x = x;
                                }
                            }
                        }

                        output.at({b, c, i, j}) = max_val;
                        if (max_y >= 0 && max_x >= 0)
                            mask.at({b, c, max_y, max_x}) = 1.0f;
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        int B = input_shape[0], C = input_shape[1];
        int H = input_shape[2], W = input_shape[3];
        int H_pad = H + 2 * pad_h;
        int W_pad = W + 2 * pad_w;
        int out_h = grad_output.shape[2];
        int out_w = grad_output.shape[3];

        Tensor grad_padded({B, C, H_pad, W_pad});
        grad_padded.fill(0.0f);

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        float grad = grad_output.at({b, c, i, j});
                        for (int m = 0; m < pool_h; ++m) {
                            for (int n = 0; n < pool_w; ++n) {
                                int y = i * stride + m;
                                int x = j * stride + n;
                                if (mask.at({b, c, y, x}) == 1.0f) {
                                    grad_padded.at({b, c, y, x}) = grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Remover padding
        Tensor grad_input({B, C, H, W});
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        grad_input.at({b, c, h, w}) = grad_padded.at({b, c, h + pad_h, w + pad_w});

        return grad_input;
    }

    size_t num_params() const override { return 0; }

    void update_weights(Optimizer* optimizer) override {}
};
