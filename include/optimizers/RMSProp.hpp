// include/optimizers/RMSProp.hpp
#pragma once

#include "Optimizer.hpp"
#include <unordered_map>
#include <cmath>

class RMSProp : public Optimizer {
private:
    float learning_rate;
    float decay_rate;
    float epsilon;
    std::unordered_map<Tensor*, Tensor> cache;

public:
    explicit RMSProp(float lr = 0.001f, float decay = 0.9f, float eps = 1e-8f)
        : learning_rate(lr), decay_rate(decay), epsilon(eps) {}

    void update(Tensor& weights, const Tensor& grads) override {
        
    }
};
