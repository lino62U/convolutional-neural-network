#pragma once

#include "Activation.hpp"
#include <cmath>

class Tanh : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        Tensor out(x.shape);
        for (int i = 0; i < x.size(); ++i) {
            out[i] = std::tanh(x[i]);
        }
        return out;
    }

    Tensor derivative(const Tensor& x) override {
        Tensor grad(x.shape);
        for (int i = 0; i < x.size(); ++i) {
            float th = std::tanh(x[i]);
            grad[i] = 1.0f - th * th; // derivada de tanh(x) es 1 - tanh(x)^2
        }
        return grad;
    }

    size_t num_params() const override {
        return 0;  // Tanh no tiene parámetros entrenables
    }

    bool is_softmax() const override {
        return false;
    }
};
