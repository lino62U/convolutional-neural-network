// include/activations/ReLU.hpp
#pragma once

#include "Activation.hpp"

class ReLU : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        Tensor out(x.shape);
        for (int i = 0; i < x.size(); ++i) {
            out[i] = std::max(0.0f, x[i]);
        }
        return out;
    }

    Tensor derivative(const Tensor& x) override {
        Tensor grad(x.shape);
        for (int i = 0; i < x.size(); ++i) {
            grad[i] = x[i] > 0 ? 1.0f : 0.0f;
        }
        return grad;
    }

    size_t num_params() const override {
        return 0;  // ReLU no tiene parámetros entrenables
    }

    // En ReLU.hpp
    bool is_softmax() const override { return false; } // Mantener por defecto
};
