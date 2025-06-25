// include/layers/MinPooling2D.hpp
#pragma once

#include "core/Layer.hpp"
#include <limits>
#include <stdexcept>

class MinPooling2D : public Layer {
private:
    int pool_h, pool_w;
    std::vector<int> input_shape;
    Tensor mask;

public:
    MinPooling2D(int pool_height, int pool_width)
        : pool_h(pool_height), pool_w(pool_width) {}

     Tensor forward(const Tensor& input, bool training = false) override {
    return Tensor();  // ⚠️ Retorno vacío por ahora
}

Tensor backward(const Tensor& grad_output) override {
    return Tensor();  // ⚠️ Retorno vacío por ahora
}


    size_t num_params() const override {
        return 0;  // Estas capas no tienen parámetros entrenables
    }
    
    void update_weights(Optimizer*) override {}
};
