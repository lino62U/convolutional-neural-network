#pragma once

#include "core/Layer.hpp"
#include "core/Initializer.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>
#include <random>

// Dropout layer
class Dropout : public Layer {
private:
    float dropout_rate;         // Probability of dropping a unit
    Tensor mask;                // Dropout mask (solo durante entrenamiento)
    std::mt19937 rng;

public:
    Dropout(float rate) : dropout_rate(rate), rng(std::random_device{}()) {
        if (rate < 0.0f || rate >= 1.0f)
            throw std::runtime_error("Dropout rate must be in [0, 1)");
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        const size_t N = input.data.size();

        if (!training) {
            // Durante evaluación, escalar por (1 - rate)
            std::vector<float> scaled_data;
            scaled_data.reserve(N);
            float scale = 1.0f - dropout_rate;
            for (float val : input.data)
                scaled_data.push_back(val * scale);

            return Tensor(std::move(scaled_data), input.shape);
        }

        // Durante entrenamiento, aplicar máscara de dropout invertido
        std::vector<float> mask_data;
        std::vector<float> dropped_data;
        mask_data.reserve(N);
        dropped_data.reserve(N);

        std::bernoulli_distribution dist(1.0f - dropout_rate);
        float scale = 1.0f / (1.0f - dropout_rate);

        for (float val : input.data) {
            float keep = dist(rng) ? scale : 0.0f;
            mask_data.push_back(keep);
            dropped_data.push_back(val * keep);
        }

        mask = Tensor(std::move(mask_data), input.shape);
        return Tensor(std::move(dropped_data), input.shape);
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape != mask.shape)
            throw std::runtime_error("Shape mismatch in Dropout::backward");

        std::vector<float> grad_input;
        grad_input.reserve(grad_output.data.size());

        for (size_t i = 0; i < grad_output.data.size(); ++i)
            grad_input.push_back(grad_output.data[i] * mask.data[i]);

        return Tensor(std::move(grad_input), grad_output.shape);
    }

    void update_weights(Optimizer*) override {
        // Dropout no tiene parámetros entrenables
    }

    size_t num_params() const override {
        return 0;
    }


    ~Dropout() override = default;
};