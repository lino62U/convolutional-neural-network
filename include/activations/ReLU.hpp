// include/activations/ReLU.hpp
#pragma once
#include "activations/Activation.hpp"
#include <algorithm>

class ReLU : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        std::vector<float> out(x.data.size());
        for (size_t i = 0; i < x.data.size(); ++i)
            out[i] = std::max(0.0f, x.data[i]);
        return Tensor(out, x.shape);
    }

    Tensor derivative(const Tensor& x) override {
        std::vector<float> grad(x.data.size());
        for (size_t i = 0; i < x.data.size(); ++i)
            grad[i] = x.data[i] > 0 ? 1.0f : 0.0f;
        return Tensor(grad, x.shape);
    }
};
