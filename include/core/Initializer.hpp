// include/core/Initializer.hpp
#pragma once

#include "Tensor.hpp"
#include <random>
#include <cmath>

class Initializer {
public:
    virtual void initialize(Tensor& tensor) = 0;
    virtual ~Initializer() {}
};

class XavierInitializer : public Initializer {
public:
    void initialize(Tensor& tensor) override {
        int fan_in = 1;
        if (tensor.shape.size() >= 2) {
            fan_in = tensor.shape[1]; // asumiendo {out_channels, in_channels, ...}
            for (size_t i = 2; i < tensor.shape.size(); ++i)
                fan_in *= tensor.shape[i];
        }

        float limit = std::sqrt(6.0f / fan_in);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (auto& val : tensor.data)
            val = dist(rng);
    }
};
