// include/core/Layer.hpp
#pragma once

#include "Tensor.hpp"
#include "optimizers/Optimizer.hpp"



// Layer base class
class Layer {
public:
    virtual Tensor forward(const Tensor& input, bool training = false) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update_weights(Optimizer* optimizer) = 0;
    virtual size_t num_params() const = 0;
    virtual ~Layer() {}
};