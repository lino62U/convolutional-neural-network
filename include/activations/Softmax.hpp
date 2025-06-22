// include/activations/Softmax.hpp
#pragma once

#include "Activation.hpp"
#include <cmath>
#include <algorithm>

class Softmax : public Activation {
public:
    Tensor activate(const Tensor& x) override {
        Tensor out(x.shape);
        float max_val = *std::max_element(x.data.begin(), x.data.end());

        float sum_exp = 0.0f;
        for (int i = 0; i < x.size(); ++i) {
            out[i] = std::exp(x[i] - max_val); // mayor estabilidad numérica
            sum_exp += out[i];
        }

        for (int i = 0; i < out.size(); ++i) {
            out[i] /= sum_exp;
        }
        return out;
    }

    // Nota: en la práctica, la derivada explícita de Softmax no se usa con CrossEntropy
    Tensor derivative(const Tensor& x) override {
        // Solo retorna el vector softmax * (1 - softmax) como una aproximación diagonal
        Tensor soft = activate(x);
        Tensor grad(x.shape);
        for (int i = 0; i < x.size(); ++i) {
            grad[i] = soft[i] * (1.0f - soft[i]);
        }
        return grad;
    }
};
