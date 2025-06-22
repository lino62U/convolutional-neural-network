// include/layers/Dense.hpp
#pragma once

#include "core/Layer.hpp"
#include "core/Initializer.hpp"
#include <random>
#include <stdexcept>
#include <numeric>
#include <functional>

class Dense : public Layer {
private:
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    Tensor grad_weights;
    Tensor grad_bias;

public:
    Dense(int input_size, int output_size, Initializer* initializer) {
        weights = Tensor({input_size, output_size});
        bias = Tensor({output_size});
        grad_weights = Tensor({input_size, output_size});
        grad_bias = Tensor({output_size});
        if (initializer) {
            initializer->initialize(weights);
        }
    }

    Tensor forward(const Tensor& input) override {
        if (input.shape.size() != 1 || input.shape[0] != weights.shape[0]) {
            throw std::invalid_argument("Dense::forward - tamaño de entrada incorrecto");
        }

        input_cache = input;
        Tensor output({weights.shape[1]}); // output_size

        for (int j = 0; j < weights.shape[1]; ++j) {
            output[j] = bias[j];
            for (int i = 0; i < weights.shape[0]; ++i) {
                output[j] += input[i] * weights[i * weights.shape[1] + j];
            }
        }
        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input({weights.shape[0]}); // input_size

        for (int i = 0; i < weights.shape[0]; ++i) {
            grad_input[i] = 0.0f;
            for (int j = 0; j < weights.shape[1]; ++j) {
                grad_input[i] += grad_output[j] * weights[i * weights.shape[1] + j];
                grad_weights[i * weights.shape[1] + j] = input_cache[i] * grad_output[j];
            }
        }

        for (int j = 0; j < weights.shape[1]; ++j) {
            grad_bias[j] = grad_output[j];
        }

        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(weights, grad_weights);
        optimizer->update(bias, grad_bias);
    }
};
