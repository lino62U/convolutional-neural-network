// include/activations/Activation.hpp
#pragma once
#include "core/Layer.hpp"

// Clase abstracta que representa una función de activación como capa
class Activation : public Layer {
protected:
    Tensor input_cache;

public:
    virtual Tensor activate(const Tensor& x) = 0;
    virtual Tensor derivative(const Tensor& x) = 0;

    Tensor forward(const Tensor& input, bool training = false) override {
        input_cache = input;
        return activate(input);
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor deriv = derivative(input_cache);
        Tensor grad_input(deriv.shape);
        for (size_t i = 0; i < deriv.data.size(); ++i) {
            grad_input.data[i] = grad_output.data[i] * deriv.data[i];
        }
        return grad_input;
    }

    void update_weights(Optimizer*) override {}
    size_t num_params() const override { return 0; }

    virtual ~Activation() = default;
};
