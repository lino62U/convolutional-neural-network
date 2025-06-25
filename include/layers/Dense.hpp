#pragma once

#include "core/Layer.hpp"
#include "core/Initializer.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>

class Dense : public Layer {
private:
    Tensor weights;       // [in_features, out_features]
    Tensor bias;          // [out_features]
    Tensor input_cache;   // [batch, in_features]
    Tensor grad_weights;  // [in_features, out_features]
    Tensor grad_bias;     // [out_features]

public:
    Dense(int in_features, int out_features, Initializer* initializer) {
        weights = Tensor({in_features, out_features});
        bias = Tensor({out_features});
        grad_weights = Tensor({in_features, out_features});
        grad_bias = Tensor({out_features});
        if (initializer) initializer->initialize(weights);
    }

    size_t num_params() const override {
        return weights.size() + bias.size();
    }

    Tensor forward(const Tensor& input) override {
        if (input.shape.size() != 2 || input.shape[1] != weights.shape[0])
            throw std::invalid_argument("Dense::forward - input must be [batch, in_features]");

        input_cache = input;

        // output = input · weights + bias (broadcast)
        Tensor output = input.dot(weights); // [batch, out_features]
        int batch = output.shape[0];
        int out_f = output.shape[1];
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < out_f; ++j) {
                output.at({b, j}) += bias.at({j});
            }
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
    int batch = grad_output.shape[0];
    int in_f = weights.shape[0];
    int out_f = weights.shape[1];

    grad_weights.fill(0.0f);
    grad_bias.fill(0.0f);

    // grad_weights = input.T · grad_output
    Tensor input_T = input_cache.transpose();                    // [in_f, batch]
    grad_weights = input_T.dot(grad_output);                     // [in_f, out_f]

    // grad_bias = suma sobre el batch
    for (int j = 0; j < out_f; ++j) {
        float sum = 0.0f;
        for (int b = 0; b < batch; ++b)
            sum += grad_output.at({b, j});
        grad_bias[j] = sum;
    }

    // grad_input = grad_output · W^T
    Tensor W_T = weights.transpose();                            // [out_f, in_f]
    Tensor grad_input = grad_output.dot(W_T);                    // [batch, in_f]

    return grad_input;
}



    void update_weights(Optimizer* optimizer) override {
        optimizer->update(weights, grad_weights);
        optimizer->update(bias, grad_bias);
    }
};
