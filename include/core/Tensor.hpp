#pragma once

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <functional>
#include <cmath>

// Tensor class
class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape; // Ej: {batch, channels, height, width}

    Tensor() = default;
    Tensor(const std::vector<float>& d, const std::vector<int>& s) : data(d), shape(s) {
        if (d.size() != std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>())) {
            throw std::runtime_error("Data size doesn't match shape");
        }
    }
    
    size_t size() const { return data.size(); }
    
    // Helper to get total elements from shape
    size_t total_elements() const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    // Basic matrix multiplication for 2D tensors
    Tensor matmul(const Tensor& other) const {
        if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0]) {
            throw std::runtime_error("Invalid shapes for matmul");
        }
        
        std::vector<float> result_data(shape[0] * other.shape[1], 0.0f);
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < other.shape[1]; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < shape[1]; ++k) {
                    sum += data[i * shape[1] + k] * other.data[k * other.shape[1] + j];
                }
                result_data[i * other.shape[1] + j] = sum;
            }
        }
        return Tensor(result_data, {shape[0], other.shape[1]});
    }

    // Element-wise addition with broadcasting support for bias
    Tensor operator+(const Tensor& other) const {
        if (shape.size() != 2 || other.shape.size() > 2) {
            throw std::runtime_error("Invalid shapes for addition");
        }

        int batch_size = shape[0];
        int feature_size = shape[1];
        int other_batch_size = other.shape.size() == 1 ? 1 : other.shape[0];
        int other_feature_size = other.shape.size() == 1 ? other.shape[0] : other.shape[1];

        if (other_batch_size != 1 || other_feature_size != feature_size) {
            throw std::runtime_error("Shape mismatch for addition with broadcasting");
        }

        std::vector<float> result_data(data.size());
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < feature_size; ++j) {
                result_data[i * feature_size + j] = data[i * feature_size + j] +
                    (other.shape.size() == 1 ? other.data[j] : other.data[j]);
            }
        }
        return Tensor(result_data, shape);
    }

    // Transpose for 2D tensor
    Tensor transpose() const {
        if (shape.size() != 2) {
            throw std::runtime_error("Transpose only supported for 2D tensors");
        }
        std::vector<float> result_data(size());
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                result_data[j * shape[0] + i] = data[i * shape[1] + j];
            }
        }
        return Tensor(result_data, {shape[1], shape[0]});
    }


    void print_shape() const {
        std::cout << "Shape: (";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i != shape.size() - 1) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    void print_matrix() const {
        if (shape.size() == 4) {
            int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    std::cout << "🖼️ Sample " << n << ", canal " << c << ":\n";
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int index = n * C * H * W + c * H * W + h * W + w;
                            std::cout << data[index] << "\t";
                        }
                        std::cout << "\n";
                    }
                }
            }
        } else if (shape.size() == 2) {
            int N = shape[0], F = shape[1];
            for (int n = 0; n < N; ++n) {
                std::cout << "🧾 Sample " << n << " (Flatten): ";
                for (int f = 0; f < F; ++f) {
                    int index = n * F + f;
                    std::cout << data[index] << " ";
                }
                std::cout << "\n";
            }
        } else if (shape.size() == 1) {
            std::cout << "📤 Vector plano: ";
            for (int i = 0; i < shape[0]; ++i) {
                std::cout << data[i] << " ";
            }
            std::cout << "\n";
        } else {
            std::cout << "⚠️  print_matrix no soporta tensores con " << shape.size() << " dimensiones.\n";
        }
    }

    Tensor slice(int start, int end) const {
        if (shape.empty() || start < 0 || end > shape[0] || start >= end) {
            throw std::runtime_error("Invalid slice range");
        }

        int batch = shape[0];
        int elements_per_sample = total_elements() / batch;

        std::vector<float> sliced_data;
        sliced_data.reserve((end - start) * elements_per_sample);

        for (int i = start; i < end; ++i) {
            sliced_data.insert(
                sliced_data.end(),
                data.begin() + i * elements_per_sample,
                data.begin() + (i + 1) * elements_per_sample
            );
        }

        std::vector<int> new_shape = shape;
        new_shape[0] = end - start;
        return Tensor(sliced_data, new_shape);
    }

        
};