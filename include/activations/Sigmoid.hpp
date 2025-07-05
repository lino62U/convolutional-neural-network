#pragma once

#include "activations/Activation.hpp"
#include <cmath>

class Sigmoid : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        std::vector<float> out(x.data.size());
        for (size_t i = 0; i < x.data.size(); ++i)
            out[i] = 1.0f / (1.0f + std::exp(-x.data[i]));
        return Tensor(out, x.shape);
    }

    Tensor derivative(const Tensor& x) override {
        std::vector<float> grad(x.data.size());
        for (size_t i = 0; i < x.data.size(); ++i) {
            float s = 1.0f / (1.0f + std::exp(-x.data[i]));
            grad[i] = s * (1.0f - s);
        }
        return Tensor(grad, x.shape);
    }
};
