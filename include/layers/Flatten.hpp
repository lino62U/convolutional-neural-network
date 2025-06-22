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
        int flat_size = 1;
        for (int dim : input.shape) {
            flat_size *= dim;
        }

        Tensor output({flat_size});
        output.data = input.data; // Copia directa
        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape);
        grad_input.data = grad_output.data; // Restaurar forma
        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {}
};
