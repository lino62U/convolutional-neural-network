// include/optimizers/SGD.hpp
#pragma once

#include "Optimizer.hpp"


// SGD Optimizer
class SGD : public Optimizer {
private:
    float learning_rate;
    float momentum;
    std::vector<Tensor> velocity; // For momentum

public:
    SGD(float lr = 0.01f, float mom = 0.9f) : learning_rate(lr), momentum(mom) {}

    void update(Tensor& param, const Tensor& grad) override {
        if (param.shape != grad.shape) {
            throw std::runtime_error("Shape mismatch in optimizer update");
        }

        // Initialize velocity if empty
        if (velocity.empty() || velocity[0].shape != param.shape) {
            velocity.clear();
            velocity.emplace_back(std::vector<float>(param.size(), 0.0f), param.shape);
        }

        // Update with momentum
        for (size_t i = 0; i < param.data.size(); ++i) {
            velocity[0].data[i] = momentum * velocity[0].data[i] - learning_rate * grad.data[i];
            param.data[i] += velocity[0].data[i];
        }
    }
};