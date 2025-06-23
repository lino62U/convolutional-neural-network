// include/core/Layer.hpp
#pragma once

#include "Tensor.hpp"

class Optimizer;

class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update_weights(Optimizer* optimizer) = 0;
    // Nuevo método para contar parámetros entrenables
    virtual size_t num_params() const = 0;
    virtual ~Layer() {}
};

