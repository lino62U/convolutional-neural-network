// include/layers/ReLUActivationLayer.hpp
#pragma once
#include "core/Layer.hpp"
#include "activations/ReLU.hpp"

class ReLUActivationLayer : public Layer {
private:
    ReLU relu;
    Tensor input_cache;
    Tensor output_cache;

public:
    Tensor forward(const Tensor& input, bool training = false) override {
        input_cache = input;
        output_cache = relu.apply_batch(input);
        return output_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input = relu.derivative_batch(input_cache, output_cache);
        for (size_t i = 0; i < grad_input.data.size(); ++i) {
            grad_input.data[i] *= grad_output.data[i];
        }
        return grad_input;
    }

    void update_weights(Optimizer*) override {}
    size_t num_params() const override { return 0; }
};