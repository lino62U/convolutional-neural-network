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
    // Constructor que recibe solo shape y rellena data con ceros
    Tensor(const std::vector<int>& s) : shape(s) {
        int total = std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>());
        data.resize(total, 0.0f); // Inicializa con ceros
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

    Tensor sum_rows() const {
        if (shape.size() != 2)
            throw std::runtime_error("sum_rows solo soporta tensores 2D");

        int rows = shape[0];
        int cols = shape[1];
        std::vector<float> result(cols, 0.0f);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result[j] += data[i * cols + j];

        return Tensor(result, {cols});
    }

    // Acceso directo a índices (batch, channel, h, w)
    float& at(int n, int c, int h, int w) {
        int idx = n * shape[1] * shape[2] * shape[3]
                + c * shape[2] * shape[3]
                + h * shape[3]
                + w;
        return data[idx];
    }

    const float& at(int n, int c, int h, int w) const {
        int idx = n * shape[1] * shape[2] * shape[3]
                + c * shape[2] * shape[3]
                + h * shape[3]
                + w;
        return data[idx];
    }
    // Acceso con vector de índices (general)
    float& at(const std::vector<int>& indices) {
        if (indices.size() != shape.size())
            throw std::runtime_error("Tensor::at - shape mismatch");
        int idx = 0;
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[idx];
    }

    const float& at(const std::vector<int>& indices) const {
        if (indices.size() != shape.size())
            throw std::runtime_error("Tensor::at - shape mismatch");
        int idx = 0;
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[idx];
    }

    // Padding con ceros (Solo para tensores 4D)
    Tensor pad(int pad) const {
        if (pad == 0) return *this;

        if (shape.size() != 4)
            throw std::runtime_error("pad solo soporta tensores 4D");

        int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
        int H_pad = H + 2 * pad;
        int W_pad = W + 2 * pad;

        Tensor result({N, C, H_pad, W_pad});
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h + pad, w + pad) = at(n, c, h, w);

        return result;
    }

    // Quitar padding (Solo para tensores 4D)
    Tensor unpad(int pad) const {
        if (pad == 0) return *this;

        if (shape.size() != 4)
            throw std::runtime_error("unpad solo soporta tensores 4D");

        int N = shape[0], C = shape[1], H = shape[2] - 2 * pad, W = shape[3] - 2 * pad;
        Tensor result({N, C, H, W});
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = at(n, c, h + pad, w + pad);

        return result;
    }

    // Rellenar con un valor específico
    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }

    // Crea tensor lleno de ceros
    static Tensor zeros(const std::vector<int>& shape_) {
        int total = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
        return Tensor(std::vector<float>(total, 0.0f), shape_);
    }

    Tensor operator*(float scalar) const {
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result_data[i] = data[i] * scalar;
        return Tensor(result_data, shape);
    }

    Tensor operator*(const Tensor& other) const {
        if (shape != other.shape)
            throw std::runtime_error("Shape mismatch in element-wise multiplication");
        
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result_data[i] = data[i] * other.data[i];
        return Tensor(result_data, shape);
    }

    Tensor softmax(int axis = -1) const {
        if (shape.size() != 2 && shape.size() != 3)
            throw std::runtime_error("Softmax solo implementado para tensores 2D o 3D");

        int dim0 = shape[0];
        int dim1 = (shape.size() >= 2 ? shape[1] : 1);
        int dim2 = (shape.size() == 3 ? shape[2] : 1);

        Tensor result(shape);
        
        if (shape.size() == 2) {
            for (int i = 0; i < dim0; ++i) {
                float max_val = -INFINITY;
                for (int j = 0; j < dim1; ++j)
                    max_val = std::max(max_val, at({i, j}));

                float sum = 0.0f;
                for (int j = 0; j < dim1; ++j) {
                    float val = std::exp(at({i, j}) - max_val);
                    result.at({i, j}) = val;
                    sum += val;
                }
                for (int j = 0; j < dim1; ++j)
                    result.at({i, j}) /= sum;
            }
        } else if (shape.size() == 3 && axis == -1) {  // Softmax sobre último eje
            for (int i = 0; i < dim0; ++i)
                for (int j = 0; j < dim1; ++j) {
                    float max_val = -INFINITY;
                    for (int k = 0; k < dim2; ++k)
                        max_val = std::max(max_val, at({i, j, k}));

                    float sum = 0.0f;
                    for (int k = 0; k < dim2; ++k) {
                        float val = std::exp(at({i, j, k}) - max_val);
                        result.at({i, j, k}) = val;
                        sum += val;
                    }

                    for (int k = 0; k < dim2; ++k)
                        result.at({i, j, k}) /= sum;
                }
        } else {
            throw std::runtime_error("Softmax: eje no soportado");
        }

        return result;
    }

    Tensor reshape(const std::vector<int>& new_shape) const {
        int new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        if (new_total != static_cast<int>(data.size()))
            throw std::runtime_error("reshape: total size mismatch");
        return Tensor(data, new_shape);
    }

    Tensor transpose(const std::vector<int>& perm) const {
        if (perm.size() != shape.size())
            throw std::runtime_error("transpose: perm dimension mismatch");

        std::vector<int> new_shape(shape.size());
        for (size_t i = 0; i < perm.size(); ++i)
            new_shape[i] = shape[perm[i]];

        // Cálculo de strides
        std::vector<int> old_strides(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; --i)
            old_strides[i] = old_strides[i + 1] * shape[i + 1];

        std::vector<int> new_strides(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; --i)
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];

        std::vector<float> new_data(data.size());

        for (size_t idx = 0; idx < data.size(); ++idx) {
            int old_idx = idx;
            std::vector<int> old_indices(shape.size(), 0);
            for (size_t i = 0; i < shape.size(); ++i) {
                old_indices[i] = old_idx / old_strides[i];
                old_idx %= old_strides[i];
            }

            std::vector<int> new_indices(shape.size());
            for (size_t i = 0; i < shape.size(); ++i)
                new_indices[i] = old_indices[perm[i]];

            int new_flat_idx = 0;
            for (size_t i = 0; i < new_shape.size(); ++i)
                new_flat_idx += new_indices[i] * new_strides[i];

            new_data[new_flat_idx] = data[idx];
        }

        return Tensor(new_data, new_shape);
    }

    Tensor dropout_mask(float drop_prob, std::mt19937& rng) const {
        std::bernoulli_distribution dist(1.0 - drop_prob);
        std::vector<float> mask_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            mask_data[i] = dist(rng) ? 1.0f : 0.0f;
        return Tensor(mask_data, shape);
    }
};