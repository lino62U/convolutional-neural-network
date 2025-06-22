#pragma once
#include "core/Layer.hpp"

class Activation : public Layer {
protected:
    Tensor input_cache;

public:
    virtual Tensor activate(const Tensor& x) = 0;
    virtual Tensor derivative(const Tensor& x) = 0;

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        return activate(input);
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor deriv = derivative(input_cache);
        Tensor grad_input;
        grad_input.shape = deriv.shape;
        grad_input.data.resize(deriv.size());

        for (int i = 0; i < deriv.size(); ++i)
            grad_input[i] = grad_output[i] * deriv[i];

        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {}

    virtual ~Activation() = default;
};
