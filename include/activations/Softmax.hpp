// include/activations/Softmax.hpp
#pragma once
#include "activations/Activation.hpp"
#include "core/Tensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

class Softmax : public Activation {
protected:
    Tensor output_cache;

public:
    Tensor activate(const Tensor& input) override {
        if (input.shape.size() != 2)
            throw std::runtime_error("Softmax expects a 2D tensor");

        const int batch_size = input.shape[0];
        const int num_classes = input.shape[1];
        std::vector<float> output(input.data.size());

        for (int i = 0; i < batch_size; ++i) {
            const int base = i * num_classes;

            // 1. Find max for numerical stability
            float max_val = *std::max_element(input.data.begin() + base, input.data.begin() + base + num_classes);

            // 2. Compute exponentials and their sum
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                output[base + j] = std::exp(input.data[base + j] - max_val);
                sum_exp += output[base + j];
            }

            // 3. Normalize
            for (int j = 0; j < num_classes; ++j) {
                output[base + j] /= sum_exp;
            }
        }

        output_cache = Tensor(output, input.shape);
        return output_cache;
    }

    Tensor derivative(const Tensor& /*input*/) override {
        // Not used — derivative handled externally (e.g., in CrossEntropyLoss)
        return Tensor(std::vector<float>(output_cache.data.size(), 1.0f), output_cache.shape);
    }
};