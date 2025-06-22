#pragma once

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <functional>
#include <cmath>

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
        if (idx >= data.size()) throw std::out_of_range("Index out of range");
        return data[idx];
    }

    const float& operator[](size_t idx) const {
        if (idx >= data.size()) throw std::out_of_range("Index out of range");
        return data[idx];
    }

    float& at(const std::vector<int>& idx) {
        return data[flatten_index(idx)];
    }

    const float& at(const std::vector<int>& idx) const {
        return data[flatten_index(idx)];
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

    Tensor slice(int start, int end) const {
        if (shape.empty()) throw std::runtime_error("Cannot slice scalar tensor");
        if (start < 0 || end > shape[0] || start >= end)
            throw std::out_of_range("Invalid slice range");

        int slice_size = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            slice_size *= shape[i];
        }

        Tensor sliced;
        sliced.shape = shape;
        sliced.shape[0] = end - start;
        sliced.data.resize((end - start) * slice_size);

        std::copy(
            data.begin() + start * slice_size,
            data.begin() + end * slice_size,
            sliced.data.begin()
        );

        return sliced;
    }

    Tensor reshape(const std::vector<int>& new_shape) const {
        int new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        if (new_total != data.size()) {
            throw std::invalid_argument("Reshape size mismatch");
        }
        Tensor reshaped;
        reshaped.data = data;
        reshaped.shape = new_shape;
        return reshaped;
    }

    static Tensor concatenate(const std::vector<Tensor>& tensors, int axis) {
        if (tensors.empty()) throw std::invalid_argument("No tensors to concatenate");

        std::vector<int> base_shape = tensors[0].shape;
        int concat_dim = 0;
        for (const auto& t : tensors) {
            if (t.shape.size() != base_shape.size())
                throw std::invalid_argument("Tensors must have same number of dimensions");
            for (size_t i = 0; i < t.shape.size(); ++i) {
                if (i == axis) continue;
                if (t.shape[i] != base_shape[i])
                    throw std::invalid_argument("Tensors must have same shape except in concatenation axis");
            }
            concat_dim += t.shape[axis];
        }

        std::vector<int> new_shape = base_shape;
        new_shape[axis] = concat_dim;
        Tensor result(new_shape);

        int offset = 0;
        for (const auto& t : tensors) {
            std::copy(t.data.begin(), t.data.end(), result.data.begin() + offset);
            offset += t.data.size();
        }
        return result;
    }

    Tensor operator+(const Tensor& other) const {
        check_same_shape(*this, other);
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        check_same_shape(*this, other);
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    Tensor operator*(float scalar) const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] * scalar;
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        check_same_shape(*this, other);
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] * other.data[i];
        return result;
    }

    // En Tensor.hpp
    Tensor dot(const Tensor& other) const {
        if (shape.size() != 2 || other.shape.size() != 2)
            throw std::invalid_argument("dot() only supports 2D tensors");

        int m = shape[0];         // filas de A
        int n = shape[1];         // columnas de A
        int p = other.shape[1];   // columnas de B

        if (n != other.shape[0])
            throw std::invalid_argument("dot(): shape mismatch");

        Tensor result({m, p});
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += this->at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }
        return result;
    }
 
    Tensor transpose() const {
        if (shape.size() != 2)
            throw std::invalid_argument("transpose() only supports 2D tensors");

        int rows = shape[0];
        int cols = shape[1];
        Tensor transposed({cols, rows}); // invierte filas y columnas

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transposed.at({j, i}) = this->at({i, j});
            }
        }
        return transposed;
    }

private:
    int flatten_index(const std::vector<int>& idx) const {
        if (idx.size() != shape.size())
            throw std::invalid_argument("Dimensionality mismatch in index");

        int index = 0;
        int stride = 1;
        for (int d = shape.size() - 1; d >= 0; --d) {
            if (idx[d] >= shape[d])
                throw std::out_of_range("Index exceeds shape dimension");
            index += idx[d] * stride;
            stride *= shape[d];
        }
        return index;
    }

    void check_same_shape(const Tensor& a, const Tensor& b) const {
        if (a.shape != b.shape) throw std::invalid_argument("Tensor shape mismatch");
    }
};
