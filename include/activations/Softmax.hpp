#pragma once
#include "Activation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>  // para std::transform_reduce
#include <stdexcept>


class Softmax : public Activation {
public:
    float apply(float x) const override {
        return x; // Not used directly, apply_batch handles the full operation
    }

    float derivative(float x) const override {
        return 1.0f; // Not used directly, derivative_batch handles it
    }

    Tensor apply_batch(const Tensor& input) const override {
        if (input.shape.size() != 2) {
            throw std::runtime_error("Softmax expects 2D input tensor");
        }
        int batch_size = input.shape[0];
        int num_classes = input.shape[1];
        std::vector<float> output(input.data.size());

        for (int i = 0; i < batch_size; ++i) {
            // Find max for numerical stability
            float max_val = *std::max_element(
                input.data.begin() + i * num_classes,
                input.data.begin() + (i + 1) * num_classes
            );
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                output[idx] = std::exp(input.data[idx] - max_val);
                sum_exp += output[idx];
            }
            // Normalize
            for (int j = 0; j < num_classes; ++j) {
                output[i * num_classes + j] /= sum_exp;
            }
        }
        return Tensor(output, input.shape);
    }

    Tensor derivative_batch(const Tensor& input, const Tensor& output) const override {
        // For softmax with cross-entropy, gradient is handled in CrossEntropyLoss
        return Tensor(std::vector<float>(input.data.size(), 1.0f), input.shape);
    }
};