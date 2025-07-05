#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include <memory>
#include <vector>

#include "core/Layer.hpp"

#include "activations/Activation.hpp"
#include "activations/ReLU.hpp"
#include "activations/Sigmoid.hpp"
#include "activations/Tanh.hpp"

class FeedForward : public Layer {
private:
    Tensor W1, b1, W2, b2;
    Tensor input_cache, z1_cache, act_cache, dropout_cache;
    Tensor W1_grad, b1_grad, W2_grad, b2_grad;
    std::shared_ptr<Activation> activation;
    float dropout_rate;
    std::mt19937 rng;

public:
    FeedForward(int embed_dim, int ff_dim, ActivationType act_type, float dropout_rate = 0.0f)
        : dropout_rate(dropout_rate), rng(std::random_device{}()) {

        switch (act_type) {
            case ActivationType::ReLU:    activation = std::make_shared<ReLU>(); break;
            case ActivationType::Sigmoid: activation = std::make_shared<Sigmoid>(); break;
            case ActivationType::Tanh:    activation = std::make_shared<Tanh>(); break;
            default:
                throw std::runtime_error("Unsupported activation function.");
        }

        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim));
        std::vector<float> w1_data(embed_dim * ff_dim);
        std::vector<float> w2_data(ff_dim * embed_dim);

        for (float& w : w1_data) w = dist(rng);
        for (float& w : w2_data) w = dist(rng);

        W1 = Tensor(w1_data, {embed_dim, ff_dim});
        W2 = Tensor(w2_data, {ff_dim, embed_dim});
        b1 = Tensor(std::vector<float>(ff_dim, 0.0f), {ff_dim});
        b2 = Tensor(std::vector<float>(embed_dim, 0.0f), {embed_dim});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3 || input.shape[2] != W1.shape[0]) {
            throw std::runtime_error("Invalid input shape for FeedForward");
        }

        input_cache = input;
        Tensor z1 = input.matmul(W1) + b1;
        z1_cache = z1;
        Tensor act = activation->forward(z1);
        act_cache = act;
        Tensor out = act.matmul(W2) + b2;

        if (training && dropout_rate > 0.0f) {
            std::vector<float> mask_data(out.data.size());
            std::bernoulli_distribution dist(1.0f - dropout_rate);
            for (size_t i = 0; i < out.data.size(); ++i)
                mask_data[i] = dist(rng) ? 1.0f / (1.0f - dropout_rate) : 0.0f;

            dropout_cache = Tensor(mask_data, out.shape);
            out = out * dropout_cache;
        } else if (!training && dropout_rate > 0.0f) {
            for (float& val : out.data) val *= (1.0f - dropout_rate);
        }

        return out;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_out = dropout_rate > 0.0f ? grad_output * dropout_cache : grad_output;

        int batch = grad_out.shape[0];
        int seq_len = grad_out.shape[1];
        int embed_dim = grad_out.shape[2];
        int ff_dim = W1.shape[1];

        Tensor act_flat = act_cache.reshape({batch * seq_len, ff_dim});
        Tensor grad_flat = grad_out.reshape({batch * seq_len, embed_dim});

        W2_grad = act_flat.transpose().matmul(grad_flat);

        std::vector<float> b2_grad_data(embed_dim, 0.0f);
        for (int i = 0; i < grad_flat.shape[0]; ++i)
            for (int j = 0; j < embed_dim; ++j)
                b2_grad_data[j] += grad_flat.data[i * embed_dim + j];
        b2_grad = Tensor(b2_grad_data, {embed_dim});

        Tensor grad_act = grad_flat.matmul(W2.transpose());
        Tensor act_deriv = activation->derivative(z1_cache);
        Tensor act_deriv_flat = act_deriv.reshape({batch * seq_len, ff_dim});
        Tensor grad_z1 = act_deriv_flat * grad_act;

        Tensor input_flat = input_cache.reshape({batch * seq_len, embed_dim});
        W1_grad = input_flat.transpose().matmul(grad_z1);

        std::vector<float> b1_grad_data(ff_dim, 0.0f);
        for (int i = 0; i < grad_z1.shape[0]; ++i)
            for (int j = 0; j < ff_dim; ++j)
                b1_grad_data[j] += grad_z1.data[i * ff_dim + j];
        b1_grad = Tensor(b1_grad_data, {ff_dim});

        Tensor grad_input_flat = grad_z1.matmul(W1.transpose());
        return grad_input_flat.reshape({batch, seq_len, embed_dim});
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(W1, W1_grad);
        optimizer->update(b1, b1_grad);
        optimizer->update(W2, W2_grad);
        optimizer->update(b2, b2_grad);
    }

    size_t num_params() const override {
        return W1.total_elements() + b1.size() + W2.total_elements() + b2.size();
    }

    ~FeedForward() override = default;
};
