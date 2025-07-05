

#pragma once

#include <memory>
#include <random>
#include <stdexcept>
#include "core/Layer.hpp"
#include "layers/Dense.hpp"
#include "activations/Activation.hpp"
#include "activations/ReLU.hpp"
#include "activations/Sigmoid.hpp"
#include "activations/Tanh.hpp"

#pragma once

#include "core/Layer.hpp"
#include "layers/Dense.hpp"
#include "activations/Activation.hpp"
#include "activations/ReLU.hpp"
#include "activations/Sigmoid.hpp"
#include "activations/Tanh.hpp"

#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

class FeedForward2 : public Layer {
private:
    std::shared_ptr<Dense> linear1;
    std::shared_ptr<Dense> linear2;
    std::shared_ptr<Activation> activation;

    Tensor input_cache, dropout_cache;
    float dropout_rate;
    std::mt19937 rng;

public:
    FeedForward2(int embed_dim, int ff_dim, ActivationType act_type, float dropout_rate = 0.0f)
        : dropout_rate(dropout_rate), rng(std::random_device{}()) {

        linear1 = std::make_shared<Dense>(embed_dim, ff_dim);
        linear2 = std::make_shared<Dense>(ff_dim, embed_dim);

        switch (act_type) {
            case ActivationType::ReLU:    activation = std::make_shared<ReLU>(); break;
            case ActivationType::Sigmoid: activation = std::make_shared<Sigmoid>(); break;
            case ActivationType::Tanh:    activation = std::make_shared<Tanh>(); break;
            default:
                throw std::runtime_error("Unsupported activation function.");
        }
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3)
            throw std::runtime_error("FeedForward::forward expects 3D tensor (batch, seq_len, embed_dim)");

        input_cache = input;

        // (B, S, E) -> (B*S, E)
        int B = input.shape[0];
        int S = input.shape[1];
        int E = input.shape[2];
        Tensor flat_input = input.reshape({B * S, E});

        Tensor out = linear1->forward(flat_input, training);         // Linear 1
        out = activation->forward(out);                              // Activation
        out = linear2->forward(out, training);                       // Linear 2

        if (training && dropout_rate > 0.0f) {
            dropout_cache = out.dropout_mask(dropout_rate, rng) / (1.0f - dropout_rate);
            out = out * dropout_cache;
        } else if (!training && dropout_rate > 0.0f) {
            out = out * (1.0f - dropout_rate);  // inference scale
        }

        return out.reshape({B, S, E});  // Volver a la forma original
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad = dropout_rate > 0.0f ? grad_output * dropout_cache : grad_output;

        int B = grad.shape[0];
        int S = grad.shape[1];
        int E = grad.shape[2];
        Tensor grad_flat = grad.reshape({B * S, E});

        // Backprop en reverse order
        Tensor grad2 = linear2->backward(grad_flat);
        Tensor act_grad = activation->derivative(linear1->get_net_cache()) * grad2;
        Tensor grad1 = linear1->backward(act_grad);

        return grad1.reshape({B, S, E});
    }

    void update_weights(Optimizer* optimizer) override {
        linear1->update_weights(optimizer);
        linear2->update_weights(optimizer);
    }

    size_t num_params() const override {
        return linear1->num_params() + linear2->num_params();
    }

    ~FeedForward2() override = default;
};
