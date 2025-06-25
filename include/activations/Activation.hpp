#pragma once
#include "core/Layer.hpp"

// Declaración adelantada para evitar inclusión circular
class Softmax;


// Activation function class
class Activation {
public:
    virtual float apply(float x) const = 0;
    virtual float derivative(float x) const = 0;
    virtual Tensor apply_batch(const Tensor& input) const {
        std::vector<float> output(input.data.size());
        for (size_t i = 0; i < input.data.size(); ++i) {
            output[i] = apply(input.data[i]);
        }
        return Tensor(output, input.shape);
    }
    virtual Tensor derivative_batch(const Tensor& input, const Tensor& output) const {
        std::vector<float> grad(input.data.size());
        for (size_t i = 0; i < input.data.size(); ++i) {
            grad[i] = derivative(input.data[i]);
        }
        return Tensor(grad, input.shape);
    }
    virtual ~Activation() {}
};
