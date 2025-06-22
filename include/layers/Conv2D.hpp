#pragma once

#include "core/Layer.hpp"
#include "core/Tensor.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>

class Conv2D : public Layer {
private:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    Tensor filters;
    Tensor bias;
    Tensor input_cache;
    Tensor grad_filters;
    Tensor grad_bias;

public:
    Conv2D(int in_ch, int out_ch, int k_h, int k_w)
        : in_channels(in_ch), out_channels(out_ch), kernel_h(k_h), kernel_w(k_w) {
        filters = Tensor({out_ch, in_ch, k_h, k_w});
        bias = Tensor({out_ch});
        grad_filters = Tensor({out_ch, in_ch, k_h, k_w});
        grad_bias = Tensor({out_ch});
        // Nota: Faltaría inicializar filtros (Xavier, He, etc.)
    }

    Tensor forward(const Tensor& input) override {
        input_cache = input; // guardar para backward
        // Esta implementación es solo un stub: se debe implementar la convolución real
        Tensor output; // calcular tamaño de salida correcto
        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        // Calcular gradientes con respecto a filtros y bias
        Tensor grad_input; // calcular tamaño correcto
        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(filters, grad_filters);
        optimizer->update(bias, grad_bias);
    }
};
