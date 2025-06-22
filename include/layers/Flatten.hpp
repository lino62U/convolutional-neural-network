// include/layers/Flatten.hpp
#pragma once

#include "core/Layer.hpp"
#include <numeric>
#include <stdexcept>

class Flatten : public Layer {
private:
    std::vector<int> input_shape;

public:
    Tensor forward(const Tensor& input) override {
        input_shape = input.shape;

        if (input.shape.size() < 2)
            throw std::invalid_argument("Flatten expects at least 2D input");

        int batch = input.shape[0];
        int feature_dim = 1;
        for (size_t i = 1; i < input.shape.size(); ++i) {
            feature_dim *= input.shape[i];
        }

        Tensor output({batch, feature_dim});
        output.data = input.data;
        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape);
        grad_input.data = grad_output.data;
        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {}
};
