#pragma once

#include "core/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>


class GlobalAveragePooling : public Layer {
private:
    Tensor input_cache;

public:
    GlobalAveragePooling() {}

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3) {
            throw std::runtime_error("GlobalAveragePooling expects 3D input tensor");
        }
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int embed_dim = input.shape[2];

        std::vector<float> output_data(batch_size * embed_dim, 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int d = 0; d < embed_dim; ++d) {
                float sum = 0.0f;
                for (int s = 0; s < seq_len; ++s) {
                    sum += input.data[n * seq_len * embed_dim + s * embed_dim + d];
                }
                output_data[n * embed_dim + d] = sum / seq_len;
            }
        }
        input_cache = input;
        return Tensor(output_data, std::vector<int>{batch_size, embed_dim});
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape.size() != 2 || grad_output.shape[0] != input_cache.shape[0] ||
            grad_output.shape[1] != input_cache.shape[2]) {
            throw std::runtime_error("Gradient shape mismatch in GlobalAveragePooling backward");
        }
        int batch_size = input_cache.shape[0];
        int seq_len = input_cache.shape[1];
        int embed_dim = input_cache.shape[2];

        std::vector<float> grad_input_data(batch_size * seq_len * embed_dim, 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < embed_dim; ++d) {
                    grad_input_data[n * seq_len * embed_dim + s * embed_dim + d] =
                        grad_output.data[n * embed_dim + d] / seq_len;
                }
            }
        }
        return Tensor(grad_input_data, std::vector<int>{batch_size, seq_len, embed_dim});
    }

    void update_weights(Optimizer* optimizer) override {}
    size_t num_params() const override { return 0; }
};