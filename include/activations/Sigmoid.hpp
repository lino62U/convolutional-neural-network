// include/activations/Sigmoid.hpp
#pragma once

#include "Activation.hpp"
#include <cmath>

class Sigmoid : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        Tensor out(x.shape);
        for (int i = 0; i < x.size(); ++i) {
            out[i] = 1.0f / (1.0f + std::exp(-x[i]));
        }
        return out;
    }

    Tensor derivative(const Tensor& x) override {
        Tensor sig = activate(x);  // reuse activate
        Tensor grad(x.shape);
        for (int i = 0; i < x.size(); ++i) {
            grad[i] = sig[i] * (1.0f - sig[i]);
        }
        return grad;
    }
};
