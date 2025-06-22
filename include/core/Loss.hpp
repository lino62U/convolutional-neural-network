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

class CrossEntropyWithLogits : public Loss {
public:
    float compute(const Tensor& y_true, const Tensor& logits) override {
        float loss = 0.0f;
        int batch = y_true.shape[0];
        int classes = y_true.shape[1];

        for (int i = 0; i < batch; ++i) {
            float max_logit = -1e9;
            for (int j = 0; j < classes; ++j) {
                max_logit = std::max(max_logit, logits[i * classes + j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < classes; ++j) {
                sum_exp += std::exp(logits[i * classes + j] - max_logit);
            }
            for (int j = 0; j < classes; ++j) {
                if (y_true[i * classes + j] > 0.0f) {
                    float log_softmax = logits[i * classes + j] - max_logit - std::log(sum_exp + 1e-9);
                    loss -= log_softmax;
                }
            }
        }

        return loss / batch;
    }

    Tensor gradient(const Tensor& y_true, const Tensor& logits) override {
        int batch = y_true.shape[0];
        int classes = y_true.shape[1];
        Tensor grad(logits.shape);

        for (int i = 0; i < batch; ++i) {
            float max_logit = -1e9;
            for (int j = 0; j < classes; ++j)
                max_logit = std::max(max_logit, logits[i * classes + j]);

            float sum_exp = 0.0f;
            for (int j = 0; j < classes; ++j)
                sum_exp += std::exp(logits[i * classes + j] - max_logit);

            for (int j = 0; j < classes; ++j) {
                float softmax = std::exp(logits[i * classes + j] - max_logit) / (sum_exp + 1e-9);
                grad[i * classes + j] = (softmax - y_true[i * classes + j]) / batch;
            }
        }

        return grad;
    }
};