#pragma once

#include "activations/Activation.hpp"
#include <cmath>

class Tanh : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        std::vector<float> out(x.data.size());
        for (size_t i = 0; i < x.data.size(); ++i)
            out[i] = std::tanh(x.data[i]);
        return Tensor(out, x.shape);
    }

    Tensor derivative(const Tensor& x) override {
        std::vector<float> grad(x.data.size());
        for (size_t i = 0; i < x.data.size(); ++i) {
            float t = std::tanh(x.data[i]);
            grad[i] = 1.0f - t * t;
        }
        return Tensor(grad, x.shape);
    }
};
