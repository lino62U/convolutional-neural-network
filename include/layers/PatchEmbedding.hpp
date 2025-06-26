#pragma once

#include "core/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

class PatchEmbedding : public Layer {
private:
    int patch_size, in_channels, embed_dim;
    Tensor weights, bias;
    Tensor input_cache, z_cache;
    Tensor weights_grad, bias_grad;
    std::mt19937 rng;

    void initialize_weights(int patch_dim, int embed_dim) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / patch_dim));
        std::vector<float> w_data(patch_dim * embed_dim);
        for (auto& w : w_data) w = dist(rng);
        weights = Tensor(w_data, {patch_dim, embed_dim});
        bias = Tensor(std::vector<float>(embed_dim, 0.0f), {embed_dim});
        weights_grad = Tensor::zeros({patch_dim, embed_dim});
        bias_grad = Tensor::zeros({embed_dim});
    }

public:
    PatchEmbedding(int in_channels, int patch_size, int embed_dim)
        : in_channels(in_channels), patch_size(patch_size), embed_dim(embed_dim),
          rng(std::random_device{}()) {
        if (patch_size <= 0 || in_channels <= 0 || embed_dim <= 0)
            throw std::runtime_error("Invalid PatchEmbedding params");
        int patch_dim = in_channels * patch_size * patch_size;
        initialize_weights(patch_dim, embed_dim);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 4 || input.shape[1] != in_channels)
            throw std::runtime_error("PatchEmbedding: invalid input shape");

        int B = input.shape[0], H = input.shape[2], W = input.shape[3];
        if (H % patch_size != 0 || W % patch_size != 0)
            throw std::runtime_error("Patch size must divide image dimensions");

        int num_patches_h = H / patch_size;
        int num_patches_w = W / patch_size;
        int N = num_patches_h * num_patches_w;
        int patch_dim = in_channels * patch_size * patch_size;

        Tensor patches({B * N, patch_dim});
        for (int b = 0; b < B; ++b) {
            int patch_idx = 0;
            for (int h = 0; h < H; h += patch_size) {
                for (int w = 0; w < W; w += patch_size) {
                    int flat_idx = b * N + patch_idx;
                    for (int c = 0; c < in_channels; ++c) {
                        for (int ph = 0; ph < patch_size; ++ph) {
                            for (int pw = 0; pw < patch_size; ++pw) {
                                int val = input.at(b, c, h + ph, w + pw);
                                int dim_idx = c * patch_size * patch_size + ph * patch_size + pw;
                                patches.data[flat_idx * patch_dim + dim_idx] = val;
                            }
                        }
                    }
                    ++patch_idx;
                }
            }
        }

        input_cache = patches;
        z_cache = patches.matmul(weights) + bias;
        return Tensor(z_cache.data, {B, N, embed_dim});
    }

    Tensor backward(const Tensor& grad_output) override {
        int B = grad_output.shape[0], N = grad_output.shape[1];
        int patch_dim = input_cache.shape[1];

        Tensor grad_z({B * N, embed_dim});
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n)
                for (int d = 0; d < embed_dim; ++d)
                    grad_z.data[(b * N + n) * embed_dim + d] =
                        grad_output.data[b * N * embed_dim + n * embed_dim + d];

        weights_grad = input_cache.transpose().matmul(grad_z);
        bias_grad = grad_z.sum_rows();

        Tensor grad_input = grad_z.matmul(weights.transpose());

        int num_patches_per_row = static_cast<int>(std::sqrt(N));
        int H = num_patches_per_row * patch_size;
        int W = H;

        Tensor grad_image({B, in_channels, H, W});
        for (int b = 0; b < B; ++b) {
            for (int p = 0; p < N; ++p) {
                int h0 = (p / num_patches_per_row) * patch_size;
                int w0 = (p % num_patches_per_row) * patch_size;
                for (int c = 0; c < in_channels; ++c) {
                    for (int ph = 0; ph < patch_size; ++ph) {
                        for (int pw = 0; pw < patch_size; ++pw) {
                            int dim_idx = c * patch_size * patch_size + ph * patch_size + pw;
                            grad_image.at(b, c, h0 + ph, w0 + pw) =
                                grad_input.data[(b * N + p) * patch_dim + dim_idx];
                        }
                    }
                }
            }
        }

        return grad_image;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(weights, weights_grad);
        optimizer->update(bias, bias_grad);
    }

    size_t num_params() const override {
        return weights.total_elements() + bias.total_elements();
    }

    ~PatchEmbedding() override = default;
};