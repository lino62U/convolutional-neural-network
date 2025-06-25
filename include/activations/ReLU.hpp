// include/activations/ReLU.hpp
#pragma once

#include "Activation.hpp"

class ReLU : public Activation {
public:
    float apply(float x) const override { return std::max(0.0f, x); }
    float derivative(float x) const override { return x > 0 ? 1.0f : 0.0f; }
};