// include/optimizers/Adam.hpp
#pragma once

#include "Optimizer.hpp"
#include <unordered_map>
#include <cmath>

class Adam : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    std::unordered_map<Tensor*, Tensor> m;
    std::unordered_map<Tensor*, Tensor> v;

public:
    explicit Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void update(Tensor& weights, const Tensor& grads) override {
        ++t;
        Tensor& mt = m[&weights];
        Tensor& vt = v[&weights];

        if (mt.data.empty()) {
            mt = Tensor(weights.shape);
            vt = Tensor(weights.shape);
        }

        for (int i = 0; i < weights.size(); ++i) {
            mt.data[i] = beta1 * mt.data[i] + (1 - beta1) * grads.data[i];
            vt.data[i] = beta2 * vt.data[i] + (1 - beta2) * grads.data[i] * grads.data[i];

            float m_hat = mt.data[i] / (1 - std::pow(beta1, t));
            float v_hat = vt.data[i] / (1 - std::pow(beta2, t));

            weights.data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};
