#pragma once

#include "core/Layer.hpp"
#include "core/Initializer.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>

// Dense layer
class Dense : public Layer {
private:
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    Tensor z_cache; // Pre-activation values
    Tensor activation_cache; // Post-activation values
    std::shared_ptr<Activation> activation;
    std::mt19937 rng;
    
    // Initialize weights using He initialization
    void initialize_weights(int in_size, int out_size) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_size));
        std::vector<float> w_data(in_size * out_size);
        for (float& w : w_data) {
            w = dist(rng);
        }
        weights = Tensor(w_data, {in_size, out_size});
        
        std::vector<float> b_data(out_size, 0.0f);
        bias = Tensor(b_data, {out_size});
    }

public:
    Dense(int in_size, int out_size, std::shared_ptr<Activation> act)
        : activation(act), rng(std::random_device{}()) {
        initialize_weights(in_size, out_size);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape[1] != weights.shape[0]) {
            throw std::runtime_error("Input shape mismatch");
        }
        input_cache = input;
        z_cache = input.matmul(weights) + bias;
        activation_cache = activation->apply_batch(z_cache);
        return activation_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        // Compute gradient w.r.t. pre-activation values
        Tensor grad_z = activation->derivative_batch(z_cache, activation_cache);
        for (size_t i = 0; i < grad_z.data.size(); ++i) {
            grad_z.data[i] *= grad_output.data[i];
        }

        // Compute gradients for weights and bias
        Tensor grad_weights = input_cache.transpose().matmul(grad_z);
        
        // Sum gradients across batch for bias
        std::vector<float> grad_bias_data(bias.shape[0], 0.0f);
        int batch_size = grad_z.shape[0];
        int out_size = grad_z.shape[1];
        for (int j = 0; j < out_size; ++j) {
            for (int i = 0; i < batch_size; ++i) {
                grad_bias_data[j] += grad_z.data[i * out_size + j];
            }
        }
        Tensor grad_bias(grad_bias_data, bias.shape);
        
        Tensor grad_input = grad_z.matmul(weights.transpose());

        // Store gradients for update
        weights_grad = grad_weights;
        bias_grad = grad_bias;

        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(weights, weights_grad);
        optimizer->update(bias, bias_grad);
    }

    size_t num_params() const override {
        return weights.total_elements() + bias.total_elements();
    }

private:
    Tensor weights_grad;
    Tensor bias_grad;
};