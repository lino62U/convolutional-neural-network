// include/core/Loss.hpp
#pragma once

#include "Tensor.hpp"
#include <cmath>
#include <stdexcept>


// Loss base class
class Loss {
public:
    virtual float compute(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual Tensor gradient(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual ~Loss() {}
};

// Mean Squared Error loss
class MSELoss : public Loss {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss computation");
        }
        float sum = 0.0f;
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            float diff = y_pred.data[i] - y_true.data[i];
            sum += diff * diff;
        }
        return sum / y_pred.data.size();
    }

    Tensor gradient(const Tensor& y_pred, const Tensor& y_true) override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss gradient");
        }
        std::vector<float> grad_data(y_pred.data.size());
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            grad_data[i] = 2.0f * (y_pred.data[i] - y_true.data[i]) / y_pred.data.size();
        }
        return Tensor(grad_data, y_pred.shape);
    }
};

// Categorical Cross-Entropy loss (assumes softmax output)
class CrossEntropyLoss : public Loss {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss computation");
        }
        float loss = 0.0f;
        int batch_size = y_pred.shape[0];
        int num_classes = y_pred.shape[1];

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                // Add small epsilon to avoid log(0)
                loss -= y_true.data[idx] * std::log(y_pred.data[idx] + 1e-10f);
            }
        }
        return loss / batch_size;
    }

    Tensor gradient(const Tensor& y_pred, const Tensor& y_true) override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss gradient");
        }
        // For softmax + cross-entropy, gradient is simply y_pred - y_true
        std::vector<float> grad_data(y_pred.data.size());
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            grad_data[i] = y_pred.data[i] - y_true.data[i];
        }
        return Tensor(grad_data, y_pred.shape);
    }
};