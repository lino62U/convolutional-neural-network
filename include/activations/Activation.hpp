#pragma once
#include "core/Layer.hpp"

// Declaración adelantada para evitar inclusión circular
class Softmax;

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
        // Cambiamos la verificación de tipo a un método virtual
        if (is_softmax()) {
            return grad_output; // Caso especial para Softmax
        }
        Tensor deriv = derivative(input_cache);
        Tensor grad_input(deriv.shape);
        for (int i = 0; i < deriv.size(); ++i) {
            grad_input[i] = grad_output[i] * deriv[i];
        }
        return grad_input;
    }

    virtual bool is_softmax() const { return false; }

    void update_weights(Optimizer*) override {}
    size_t num_params() const override { return 0; }
    virtual ~Activation() = default;
};