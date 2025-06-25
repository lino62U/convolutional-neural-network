// include/metrics/Metric.hpp
#pragma once

#include "Metric.hpp"
#include <stdexcept>

#include "core/Tensor.hpp"

// Metric base class
class Metric {
public:
    virtual float compute(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual std::string name() const = 0;
    virtual ~Metric() {}
};

// Accuracy metric
class Accuracy : public Metric {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) override {
        if (y_pred.shape != (y_true.shape)) {
            throw std::runtime_error("Shape mismatch in accuracy computation");
        }
        int batch_size = y_pred.shape[0];
        int num_classes = y_pred.shape[1];
        int correct = 0;

        for (int i = 0; i < batch_size; ++i) {
            // Find predicted and true class
            int pred_class = 0;
            int true_class = 0;
            float max_pred = y_pred.data[i * num_classes];
            float max_true = y_true.data[i * num_classes];

            for (int j = 1; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                if (y_pred.data[idx] > max_pred) {
                    max_pred = y_pred.data[idx];
                    pred_class = j;
                }
                if (y_true.data[idx] > max_true) {
                    max_true = y_true.data[idx];
                    true_class = j;
                }
            }
            if (pred_class == true_class) {
                ++correct;
            }
        }
        return static_cast<float>(correct) / batch_size;
    }

    std::string name() const override { return "accuracy"; }
};