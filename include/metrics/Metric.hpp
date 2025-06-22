// include/metrics/Metric.hpp
#pragma once

#include "Metric.hpp"
#include <stdexcept>

#include "core/Tensor.hpp"

class Metric {
public:
    virtual float compute(const Tensor& y_true, const Tensor& y_pred) const = 0;
    virtual ~Metric() = default;
};



class Accuracy : public Metric {
public:
    float compute(const Tensor& y_true, const Tensor& y_pred) const override {
        if (y_true.shape.size() != 2 || y_pred.shape.size() != 2)
            throw std::invalid_argument("Accuracy::compute espera tensores 2D [batch, clases]");

        int batch = y_true.shape[0];
        int correct = 0;

        for (int i = 0; i < batch; ++i) {
            int true_label = argmax(y_true, i);
            int pred_label = argmax(y_pred, i);
            if (true_label == pred_label)
                correct++;
        }

        return static_cast<float>(correct) / batch;
    }

private:
    static int argmax(const Tensor& tensor, int row) {
        int num_classes = tensor.shape[1];
        int offset = row * num_classes;
        float max_val = tensor.data[offset];
        int max_idx = 0;
        for (int j = 1; j < num_classes; ++j) {
            if (tensor.data[offset + j] > max_val) {
                max_val = tensor.data[offset + j];
                max_idx = j;
            }
        }
        return max_idx;
    }
};
