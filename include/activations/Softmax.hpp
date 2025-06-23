#pragma once
#include "Activation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>  // para std::transform_reduce
#include <stdexcept>

class Softmax : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        Tensor out(x.shape);
        const float max_val = *std::max_element(x.data.begin(), x.data.end());
        
        // Calcula sum_exp y aplica exp en un solo paso (C++17)
        const float sum_exp = std::transform_reduce(
            x.data.begin(), x.data.end(), 0.0f, std::plus{}, 
            [max_val](float val) { return std::exp(val - max_val); }
        );
        
        // Normaliza
        std::transform(x.data.begin(), x.data.end(), out.data.begin(),
            [sum_exp, max_val](float val) { return std::exp(val - max_val) / sum_exp; });
        
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