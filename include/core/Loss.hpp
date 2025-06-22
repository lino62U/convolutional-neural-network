// include/core/Loss.hpp
#pragma once

#include "Tensor.hpp"
#include <cmath>
#include <stdexcept>

class Loss {
public:
    virtual float compute(const Tensor& y_true, const Tensor& y_pred) = 0;
    virtual Tensor gradient(const Tensor& y_true, const Tensor& y_pred) = 0;
    virtual ~Loss() {}
};

class CrossEntropyLoss : public Loss {
public:
    float compute(const Tensor& y_true, const Tensor& y_pred) override {
        if (y_true.size() != y_pred.size())
            throw std::invalid_argument("CrossEntropyLoss::compute: tamaño incompatible.");

        float loss = 0.0f;
        for (size_t i = 0; i < y_true.data.size(); ++i) {
            float p = std::max(std::min(y_pred.data[i], 1.0f - 1e-7f), 1e-7f);  // Evitar log(0)
            loss -= y_true.data[i] * std::log(p);
        }
        return loss / static_cast<float>(y_true.data.size());
    }

    Tensor gradient(const Tensor& y_true, const Tensor& y_pred) override {
        if (y_true.size() != y_pred.size())
            throw std::invalid_argument("CrossEntropyLoss::gradient: tamaño incompatible.");

        Tensor grad(y_pred.shape);
        for (size_t i = 0; i < y_true.data.size(); ++i) {
            float p = std::max(std::min(y_pred.data[i], 1.0f - 1e-7f), 1e-7f);  // Evitar división por 0
            grad.data[i] = -y_true.data[i] / p;
        }
        return grad;
    }
};
