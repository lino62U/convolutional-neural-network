// include/layers/Flatten.hpp
#pragma once

#include "core/Layer.hpp"
#include <numeric>
#include <stdexcept>


// Flatten layer
class Flatten : public Layer {
public:
    Flatten() {}

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() < 2) {
            throw std::runtime_error("Flatten expects at least 2D input tensor");
        }
        int batch_size = input.shape[0];
        int flat_size = std::accumulate(input.shape.begin() + 1, input.shape.end(), 1, std::multiplies<int>());
        input_cache = input;
        return Tensor(input.data, {batch_size, flat_size});
    }

    Tensor backward(const Tensor& grad_output) override {
        return Tensor(grad_output.data, input_cache.shape);
    }

    void update_weights(Optimizer* optimizer) override {
        // No trainable parameters
    }

    size_t num_params() const override {
        return 0;
    }

private:
    Tensor input_cache;
};