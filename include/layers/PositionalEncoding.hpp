#pragma once

#include "core/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

class PositionalEncoding : public Layer {
private:
    Tensor encodings;
    Tensor encodings_grad;
    bool learnable;

public:
    PositionalEncoding(int num_patches, int embed_dim, bool learnable = false)
        : learnable(learnable) {
        std::vector<float> data(num_patches * embed_dim);

        if (!learnable) {
            for (int pos = 0; pos < num_patches; ++pos) {
                for (int i = 0; i < embed_dim; ++i) {
                    float angle = pos / std::pow(10000.0f, i / float(embed_dim));
                    data[pos * embed_dim + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
                }
            }
        } else {
            std::mt19937 rng(std::random_device{}());
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (float& x : data) x = dist(rng);
        }

        encodings = Tensor(data, {num_patches, embed_dim});
        encodings_grad = Tensor::zeros({num_patches, embed_dim});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3)
            throw std::runtime_error("PositionalEncoding: expected input shape (B, N, D)");

        int B = input.shape[0];
        int N = input.shape[1];
        int D = input.shape[2];

        if (encodings.shape[0] != N || encodings.shape[1] != D)
            throw std::runtime_error("PositionalEncoding: encoding shape mismatch");

        Tensor output(input.shape);
        for (int b = 0; b < B; ++b) {
            for (int p = 0; p < N; ++p) {
                for (int d = 0; d < D; ++d) {
                    output.at(b, p, d) = input.at(b, p, d) + encodings.at(p, d);
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        if (!learnable) return grad_output;

        int B = grad_output.shape[0];
        int N = grad_output.shape[1];
        int D = grad_output.shape[2];

        for (int p = 0; p < N; ++p) {
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int b = 0; b < B; ++b) {
                    sum += grad_output.at(b, p, d);
                }
                encodings_grad.at(p, d) = sum;
            }
        }

        return grad_output; // Grad propagates unchanged
    }

    void update_weights(Optimizer* optimizer) override {
        if (learnable) {
            optimizer->update(encodings, encodings_grad);
        }
    }

    size_t num_params() const override {
        return learnable ? encodings.total_elements() : 0;
    }
};
