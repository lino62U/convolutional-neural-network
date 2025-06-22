// include/core/Tensor.hpp
#pragma once

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape; // Ej: {batch, channels, height, width}

    Tensor() = default;

    Tensor(const std::vector<int>& shape_) : shape(shape_) {
        int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        data.resize(total, 0.0f);
    }

    float& operator[](size_t idx) {
        if (idx >= data.size()) throw std::out_of_range("Tensor index out of range");
        return data[idx];
    }

    const float& operator[](size_t idx) const {
        if (idx >= data.size()) throw std::out_of_range("Tensor index out of range");
        return data[idx];
    }

    int size() const {
        return static_cast<int>(data.size());
    }

    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }

    void print(const std::string& label = "Tensor") const {
        std::cout << label << " [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << "x";
        }
        std::cout << "]\n";
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i] << " ";
            if ((i + 1) % shape.back() == 0) std::cout << "\n";
        }
        std::cout << std::endl;
    }
};
