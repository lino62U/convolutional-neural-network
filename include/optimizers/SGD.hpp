// include/optimizers/SGD.hpp
#pragma once

#include "Optimizer.hpp"

class SGD : public Optimizer {
private:
    float learning_rate;

public:
    explicit SGD(float lr = 0.01f) : learning_rate(lr) {}

    void update(Tensor& weights, const Tensor& grads) override {
        for (int i = 0; i < weights.size(); ++i) {
            weights.data[i] -= learning_rate * grads.data[i];
        }
    }
};
