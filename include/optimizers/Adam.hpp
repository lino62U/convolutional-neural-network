// include/optimizers/Adam.hpp
#pragma once

#include "Optimizer.hpp"
#include <unordered_map>
#include <cmath>


// Adam Optimizer
class Adam : public Optimizer {
private:
    float learning_rate;
    float beta1; // Decay rate for first moment
    float beta2; // Decay rate for second moment
    float epsilon; // For numerical stability
    int t; // Timestep
    std::vector<Tensor> m; // First moment (mean)
    std::vector<Tensor> v; // Second moment (uncentered variance)

public:
    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void update(Tensor& param, const Tensor& grad) override {
        if (param.shape != grad.shape) {
            throw std::runtime_error("Shape mismatch in optimizer update");
        }

        // Initialize moment vectors if empty or shape mismatch
        if (m.empty() || m[0].shape != param.shape) {
            m.clear();
            v.clear();
            m.emplace_back(std::vector<float>(param.size(), 0.0f), param.shape);
            v.emplace_back(std::vector<float>(param.size(), 0.0f), param.shape);
        }

        // Increment timestep
        ++t;

        // Update biased first and second moments
        for (size_t i = 0; i < param.data.size(); ++i) {
            m[0].data[i] = beta1 * m[0].data[i] + (1.0f - beta1) * grad.data[i];
            v[0].data[i] = beta2 * v[0].data[i] + (1.0f - beta2) * grad.data[i] * grad.data[i];
        }

        // Compute bias-corrected moments
        std::vector<float> m_hat(param.size()), v_hat(param.size());
        float beta1_t = std::pow(beta1, t);
        float beta2_t = std::pow(beta2, t);
        for (size_t i = 0; i < param.data.size(); ++i) {
            m_hat[i] = m[0].data[i] / (1.0f - beta1_t);
            v_hat[i] = v[0].data[i] / (1.0f - beta2_t);
            param.data[i] -= learning_rate * m_hat[i] / (std::sqrt(v_hat[i]) + epsilon);
        }
    }
};
