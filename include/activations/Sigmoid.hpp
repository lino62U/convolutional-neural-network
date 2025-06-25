// include/activations/Sigmoid.hpp
#pragma once

#include "Activation.hpp"
#include <cmath>

class Sigmoid : public Activation {
public:
    float apply(float x) const override { return 1.0f / (1.0f + std::exp(-x)); }
    float derivative(float x) const override {
        float s = apply(x);
        return s * (1.0f - s);
    }
};