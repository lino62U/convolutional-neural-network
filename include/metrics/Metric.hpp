// include/metrics/Metric.hpp
#pragma once

#include "Metric.hpp"
#include <stdexcept>

#include "core/Tensor.hpp"

/* ------------------------------------------------------------------
 * Utilidad común: precision, recall y f1 (forward declaration + body)
 * ------------------------------------------------------------------*/
inline float compute_precision_recall_f1(const Tensor& y_pred,
                                         const Tensor& y_true,
                                         const std::string& mode);

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

// Precision
class Precision : public Metric {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) override {
        return compute_precision_recall_f1(y_pred, y_true, "precision");
    }

    std::string name() const override { return "precision"; }
};

// Recall
class Recall : public Metric {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) override {
        return compute_precision_recall_f1(y_pred, y_true, "recall");
    }

    std::string name() const override { return "recall"; }
};

// F1 Score
class F1Score : public Metric {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) override {
        return compute_precision_recall_f1(y_pred, y_true, "f1");
    }

    std::string name() const override { return "f1"; }
};

// Utility function for precision, recall, f1
inline float compute_precision_recall_f1(const Tensor& y_pred, const Tensor& y_true, const std::string& mode) {
    if (y_pred.shape != y_true.shape) {
        throw std::runtime_error("Shape mismatch in metric computation");
    }

    int batch_size = y_pred.shape[0];
    int num_classes = y_pred.shape[1];

    std::vector<int> true_labels(batch_size), pred_labels(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        int pred_class = 0, true_class = 0;
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

        pred_labels[i] = pred_class;
        true_labels[i] = true_class;
    }

    int TP = 0, FP = 0, FN = 0;

    for (int c = 0; c < num_classes; ++c) {
        for (int i = 0; i < batch_size; ++i) {
            bool pred_is_c = pred_labels[i] == c;
            bool true_is_c = true_labels[i] == c;

            if (pred_is_c && true_is_c) TP++;
            else if (pred_is_c && !true_is_c) FP++;
            else if (!pred_is_c && true_is_c) FN++;
        }
    }

    float precision = (TP + FP > 0) ? static_cast<float>(TP) / (TP + FP) : 0.0f;
    float recall    = (TP + FN > 0) ? static_cast<float>(TP) / (TP + FN) : 0.0f;
    float f1        = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;

    if (mode == "precision") return precision;
    if (mode == "recall") return recall;
    return f1;
}