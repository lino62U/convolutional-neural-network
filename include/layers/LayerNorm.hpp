#pragma once

#include "core/Layer.hpp"
#include <limits>
#include <stdexcept>
#include <cmath>



class LayerNorm : public Layer {
private:
    float epsilon;
    Tensor gamma, beta;
    Tensor gamma_grad, beta_grad; // Added gradient tensors
    Tensor input_cache;

public:
    LayerNorm(int dim, float eps = 1e-5f) : epsilon(eps) {
        std::vector<float> gamma_data(dim, 1.0f);
        std::vector<float> beta_data(dim, 0.0f);
        std::vector<float> grad_data(dim, 0.0f); // Initialize gradients to zero
        gamma = Tensor(gamma_data, std::vector<int>{dim});
        beta = Tensor(beta_data, std::vector<int>{dim});
        gamma_grad = Tensor(grad_data, std::vector<int>{dim});
        beta_grad = Tensor(grad_data, std::vector<int>{dim});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3) {
            throw std::runtime_error("LayerNorm expects 3D input tensor");
        }
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int dim = input.shape[2];
        if (dim != gamma.shape[0]) {
            throw std::runtime_error("LayerNorm dimension mismatch");
        }

        std::vector<float> output_data(input.data.size());
        for (int n = 0; n < batch_size; ++n) {
            for (int s = 0; s < seq_len; ++s) {
                // Compute mean
                float mean = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    mean += input.data[n * seq_len * dim + s * dim + d];
                }
                mean /= dim;

                // Compute variance
                float var = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = input.data[n * seq_len * dim + s * dim + d] - mean;
                    var += diff * diff;
                }
                var /= dim;
                var = std::sqrt(var + epsilon);

                // Normalize and scale
                for (int d = 0; d < dim; ++d) {
                    int idx = n * seq_len * dim + s * dim + d;
                    output_data[idx] = (input.data[idx] - mean) / var * gamma.data[d] + beta.data[d];
                }
            }
        }
        input_cache = input;
        return Tensor(output_data, input.shape);
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape != input_cache.shape) {
            throw std::runtime_error("Gradient shape mismatch in LayerNorm backward");
        }
        int batch_size = input_cache.shape[0];
        int seq_len = input_cache.shape[1];
        int dim = input_cache.shape[2];

        std::vector<float> grad_input_data(input_cache.data.size(), 0.0f);
        std::vector<float> grad_gamma_data(dim, 0.0f);
        std::vector<float> grad_beta_data(dim, 0.0f);

        for (int n = 0; n < batch_size; ++n) {
            for (int s = 0; s < seq_len; ++s) {
                // Recompute mean and variance
                float mean = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    mean += input_cache.data[n * seq_len * dim + s * dim + d];
                }
                mean /= dim;

                float var = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = input_cache.data[n * seq_len * dim + s * dim + d] - mean;
                    var += diff * diff;
                }
                var /= dim;
                var = std::sqrt(var + epsilon);

                // Compute normalized input
                std::vector<float> x_hat(dim);
                for (int d = 0; d < dim; ++d) {
                    x_hat[d] = (input_cache.data[n * seq_len * dim + s * dim + d] - mean) / var;
                }

                // Gradients for gamma and beta
                for (int d = 0; d < dim; ++d) {
                    grad_gamma_data[d] += grad_output.data[n * seq_len * dim + s * dim + d] * x_hat[d];
                    grad_beta_data[d] += grad_output.data[n * seq_len * dim + s * dim + d];
                }

                // Gradient for input
                float sum_grad = 0.0f;
                float sum_grad_x_hat = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    sum_grad += grad_output.data[n * seq_len * dim + s * dim + d] * gamma.data[d];
                    sum_grad_x_hat += grad_output.data[n * seq_len * dim + s * dim + d] * gamma.data[d] * x_hat[d];
                }
                for (int d = 0; d < dim; ++d) {
                    int idx = n * seq_len * dim + s * dim + d;
                    grad_input_data[idx] = (grad_output.data[idx] * gamma.data[d] - sum_grad / dim -
                                           x_hat[d] * sum_grad_x_hat / dim) / var;
                }
            }
        }

        gamma_grad = Tensor(grad_gamma_data, std::vector<int>{dim});
        beta_grad = Tensor(grad_beta_data, std::vector<int>{dim});
        return Tensor(grad_input_data, input_cache.shape);
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(gamma, gamma_grad);
        optimizer->update(beta, beta_grad);
    }

    size_t num_params() const override {
        return gamma.data.size() + beta.data.size();
    }
};