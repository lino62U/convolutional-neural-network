// include/optimizers/Optimizer.hpp
#pragma once

#include "core/Tensor.hpp"


class Optimizer {
public:
    virtual void update(Tensor& weights, const Tensor& grads) = 0;
    virtual ~Optimizer() {}
};