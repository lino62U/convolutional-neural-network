#pragma once

#include "core/Layer.hpp"
#include "core/Initializer.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>


// Dropout layer
class Dropout : public Layer {
private:
    float dropout_rate; // Probability of dropping a unit (p)
    Tensor mask; // Dropout mask for backpropagation
    std::mt19937 rng;

public:
    Dropout(float rate) : dropout_rate(rate), rng(std::random_device{}()) {
        if (rate < 0.0f || rate >= 1.0f) {
            throw std::runtime_error("Dropout rate must be in [0, 1)");
        }
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (!training) {
            // During evaluation, scale by (1 - dropout_rate)
            std::vector<float> output(input.data.size());
            for (size_t i = 0; i < input.data.size(); ++i) {
                output[i] = input.data[i] * (1.0f - dropout_rate);
            }
            return Tensor(output, input.shape);
        }

        // During training, generate random mask and apply dropout
        std::vector<float> mask_data(input.data.size());
        std::vector<float> output(input.data.size());
        std::bernoulli_distribution dist(1.0f - dropout_rate); // Keep probability
        for (size_t i = 0; i < input.data.size(); ++i) {
            mask_data[i] = dist(rng) ? 1.0f / (1.0f - dropout_rate) : 0.0f; // Inverted dropout
            output[i] = input.data[i] * mask_data[i];
        }
        mask = Tensor(mask_data, input.shape);
        return Tensor(output, input.shape);
    }

    Tensor backward(const Tensor& grad_output) override {
        // Apply mask to gradients
        if (grad_output.shape != mask.shape) {
            throw std::runtime_error("Shape mismatch in dropout backward");
        }
        std::vector<float> grad_input(grad_output.data.size());
        for (size_t i = 0; i < grad_output.data.size(); ++i) {
            grad_input[i] = grad_output.data[i] * mask.data[i];
        }
        return Tensor(grad_input, grad_output.shape);
    }

    void update_weights(Optimizer* optimizer) override {
        // No trainable parameters
    }

    size_t num_params() const override {
        return 0;
    }
};