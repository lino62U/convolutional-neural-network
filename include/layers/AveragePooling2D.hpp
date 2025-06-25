#pragma once

#include "core/Layer.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

class AveragePooling2D : public Layer {
private:
    int pool_h, pool_w;
    int stride;
    std::string padding_type;
    int pad_h = 0, pad_w = 0;
    std::vector<int> input_shape;

public:
    AveragePooling2D(int pool_height, int pool_width, int stride_ = 1, const std::string& padding = "valid")
        : pool_h(pool_height), pool_w(pool_width),
          stride(stride_), padding_type(padding) {}
Tensor forward(const Tensor& input, bool training = false) override {
    return Tensor();  // ⚠️ Retorno vacío por ahora
}

Tensor backward(const Tensor& grad_output) override {
    return Tensor();  // ⚠️ Retorno vacío por ahora
}


    size_t num_params() const override { return 0; }

    void update_weights(Optimizer*) override {}
};
