#pragma once

#include "core/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include <random>
#include <stdexcept>

class Dense : public Layer {
private:
    Tensor W;                  // Pesos
    Tensor b;                  // Bias
    Tensor dW;                 // Gradiente de pesos
    Tensor db;                 // Gradiente de bias

    Tensor input_cache;        // X
    Tensor net_cache;          // XW + b

    std::mt19937 rng;

    void initialize_weights(int in_features, int out_features) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features));
        std::vector<float> w_data(in_features * out_features);
        for (auto& w : w_data)
            w = dist(rng);
        W = Tensor(w_data, {in_features, out_features});
        b = Tensor(std::vector<float>(out_features, 0.0f), {out_features});
    }

public:
    Dense(int in_features, int out_features)
        : rng(std::random_device{}()) {
        initialize_weights(in_features, out_features);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape[1] != W.shape[0])
            throw std::runtime_error("Dense: Input shape mismatch");

        input_cache = input;                                // Cache X
        net_cache = input.matmul(W) + b;                    // Z = XW + b
        return Tensor(std::move(net_cache));                // Retorna moviendo (sin copia)
    }

    Tensor backward(const Tensor& grad_output) override {
        dW = input_cache.transpose().matmul(grad_output);   // dW = Xᵀ·dZ
        db = grad_output.sum_rows();                        // db = sum over batch
        return grad_output.matmul(W.transpose());           // dX = dZ·Wᵀ
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(W, dW);
        optimizer->update(b, db);
    }

    size_t num_params() const override {
        return W.total_elements() + b.total_elements();
    }

    ~Dense() override = default;
};
