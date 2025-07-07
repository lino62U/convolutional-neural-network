// include/optimizers/Optimizer.hpp
#pragma once

#include "core/Tensor.hpp"


// Optimizer base class
class Optimizer {
public:
    virtual void update(Tensor& param, const Tensor& grad) = 0;
    virtual ~Optimizer() {}
};