#pragma once
#include "Activation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>  // para std::transform_reduce
#include <stdexcept>

class Softmax : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        if (x.shape.size() != 2)
            throw std::invalid_argument("Softmax espera tensor 2D [batch, clases]");

        Tensor out(x.shape);
        int batch = x.shape[0];
        int num_classes = x.shape[1];

        for (int b = 0; b < batch; ++b) {
            float max_val = -1e9;
            for (int j = 0; j < num_classes; ++j)
                max_val = std::max(max_val, x.at({b, j}));

            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j)
                sum_exp += std::exp(x.at({b, j}) - max_val);

            for (int j = 0; j < num_classes; ++j)
                out.at({b, j}) = std::exp(x.at({b, j}) - max_val) / (sum_exp + 1e-9f);
        }

        return out;
    }


    Tensor derivative(const Tensor& x) override {
        // Este método no debería usarse cuando se combina con CrossEntropyLoss
        throw std::runtime_error(
            "No usar derivative() de Softmax directamente. "
            "Usar con CrossEntropyLoss para gradientes automáticos."
        );
    }

    size_t num_params() const override {
        return 0;  // ReLU no tiene parámetros entrenables
    }

    bool is_softmax() const override { return true; } // Identificador especial
};