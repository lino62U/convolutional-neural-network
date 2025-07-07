#pragma once

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <functional>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <omp.h>
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
Tensor matmul(const Tensor& other) const {
    // 2D x 2D => (M, K) x (K, N) = (M, N)
    if (shape.size() == 2 && other.shape.size() == 2) {
        int M = shape[0];
        int K = shape[1];
        int N = other.shape[1];
        if (K != other.shape[0])
            throw std::runtime_error("Shape mismatch in 2D matmul");

        std::vector<float> result_data(M * N, 0.0f);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += data[i * K + k] * other.data[k * N + j];
                }
                result_data[i * N + j] = sum;
            }
        }

        return Tensor(result_data, {M, N});
    }

    // 3D x 3D => (B, M, K) x (B, K, N) = (B, M, N)
    if (shape.size() == 3 && other.shape.size() == 3) {
        int B = shape[0];
        int M = shape[1];
        int K = shape[2];
        int K2 = other.shape[1];
        int N = other.shape[2];
        if (B != other.shape[0] || K != K2)
            throw std::runtime_error("Shape mismatch in 3D batched matmul");

        std::vector<float> result_data(B * M * N, 0.0f);

        #pragma omp parallel for collapse(3)
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += data[b * M * K + i * K + k] * other.data[b * K * N + k * N + j];
                    }
                    result_data[b * M * N + i * N + j] = sum;
                }
            }
        }

        return Tensor(result_data, {B, M, N});
    }

    // 3D x 2D => (B, M, K) x (K, N) = (B, M, N)
    if (shape.size() == 3 && other.shape.size() == 2) {
        int B = shape[0];
        int M = shape[1];
        int K = shape[2];
        int K2 = other.shape[0];
        int N = other.shape[1];
        if (K != K2)
            throw std::runtime_error("Shape mismatch in 3D x 2D matmul");

        std::vector<float> result_data(B * M * N, 0.0f);

        #pragma omp parallel for collapse(3)
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += data[b * M * K + i * K + k] * other.data[k * N + j];
                    }
                    result_data[b * M * N + i * N + j] = sum;
                }
            }
        }

        return Tensor(result_data, {B, M, N});
    }

    throw std::runtime_error("Unsupported shapes for matmul: left shape = " + shape_str() + ", right shape = " + other.shape_str());
}
    // Element-wise addition with broadcasting support for bias
Tensor operator+(const Tensor& other) const {
    if (shape == other.shape) {
        // Mismo shape, suma directa
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result_data[i] = data[i] + other.data[i];
        return Tensor(result_data, shape);
    }

    // Broadcasting: (B, D) + (D)
    if (shape.size() == 2 && other.shape.size() == 1 && shape[1] == other.shape[0]) {
        int B = shape[0], D = shape[1];
        std::vector<float> result_data(data.size());
        for (int i = 0; i < B; ++i)
            for (int j = 0; j < D; ++j)
                result_data[i * D + j] = data[i * D + j] + other.data[j];
        return Tensor(result_data, shape);
    }

    // Broadcasting: (B, N, D) + (D)
    if (shape.size() == 3 && other.shape.size() == 1 && shape[2] == other.shape[0]) {
        int B = shape[0], N = shape[1], D = shape[2];
        std::vector<float> result_data(data.size());
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n)
                for (int d = 0; d < D; ++d)
                    result_data[b * N * D + n * D + d] =
                        data[b * N * D + n * D + d] + other.data[d];
        return Tensor(result_data, shape);
    }

    // Broadcasting: (B, N, D) + (1, 1, D)
    if (shape.size() == 3 && other.shape.size() == 3 &&
        other.shape[0] == 1 && other.shape[1] == 1 && other.shape[2] == shape[2]) {
        int B = shape[0], N = shape[1], D = shape[2];
        std::vector<float> result_data(data.size());
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n)
                for (int d = 0; d < D; ++d)
                    result_data[b * N * D + n * D + d] =
                        data[b * N * D + n * D + d] + other.data[d];
        return Tensor(result_data, shape);
    }

    throw std::runtime_error("Shape mismatch in Tensor::operator+");
}


    // Transpose for 2D tensor
    
    


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

        #pragma omp parallel for collapse(4)
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

        #pragma omp parallel for collapse(4)
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
    if (shape.size() == 2) {
        int rows = shape[0], cols = shape[1];
        Tensor result(shape);
        for (int i = 0; i < rows; ++i) {
            float max_val = -INFINITY;
            for (int j = 0; j < cols; ++j)
                max_val = std::max(max_val, at({i, j}));

            float sum = 0.0f;
            for (int j = 0; j < cols; ++j) {
                float val = std::exp(at({i, j}) - max_val);
                result.at({i, j}) = val;
                sum += val;
            }
            for (int j = 0; j < cols; ++j)
                result.at({i, j}) /= sum;
        }
        return result;
    }

    if (shape.size() == 3 && axis == -1) {
        int B = shape[0], T = shape[1], D = shape[2];
        Tensor result(shape);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                float max_val = -INFINITY;
                for (int d = 0; d < D; ++d)
                    max_val = std::max(max_val, at({b, t, d}));

                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float val = std::exp(at({b, t, d}) - max_val);
                    result.at({b, t, d}) = val;
                    sum += val;
                }
                for (int d = 0; d < D; ++d)
                    result.at({b, t, d}) /= sum;
            }
        }
        return result;
    }

    if (shape.size() == 4) {
        int B = shape[0], H = shape[1], T = shape[2], D = shape[3];
        Tensor result(shape);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int t = 0; t < T; ++t) {
                    float max_val = -INFINITY;
                    for (int d = 0; d < D; ++d)
                        max_val = std::max(max_val, at({b, h, t, d}));

                    float sum = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        float val = std::exp(at({b, h, t, d}) - max_val);
                        result.at({b, h, t, d}) = val;
                        sum += val;
                    }
                    for (int d = 0; d < D; ++d)
                        result.at({b, h, t, d}) /= sum;
                }
            }
        }
        return result;
    }

    throw std::runtime_error("Softmax solo implementado para tensores 2D, 3D o 4D (axis=-1), recibí shape: " + shape_str());
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
    
    /*
    Tensor transpose() const {
        if (shape.size() < 2)
            throw std::runtime_error("Transpose requires at least 2D tensor");
        return transpose(shape.size() - 2, shape.size() - 1);
    }
    */
    
    Tensor transpose(int dim0 = -1, int dim1 = -1) const {
        if (shape.size() < 2 || shape.size() > 4) {
            throw std::runtime_error("Transpose supported for 2D, 3D, or 4D tensors only");
        }

        if (dim0 == -1 && dim1 == -1) {
            dim0 = shape.size() - 2;
            dim1 = shape.size() - 1;
        }
        if (dim0 < 0 || dim1 < 0 || dim0 >= (int)shape.size() || dim1 >= (int)shape.size() || dim0 == dim1) {
            throw std::runtime_error("Invalid transpose dimensions");
        }

        std::vector<int> new_shape = shape;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::vector<float> new_data(size(), 0.0f);

        if (shape.size() == 2) {
            int rows = shape[0];
            int cols = shape[1];

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    new_data[j * rows + i] = data[i * cols + j];
                }
            }

        } else if (shape.size() == 3) {
            int d0 = shape[0], d1 = shape[1], d2 = shape[2];

            #pragma omp parallel for collapse(3)
            for (int i = 0; i < d0; ++i) {
                for (int j = 0; j < d1; ++j) {
                    for (int k = 0; k < d2; ++k) {
                        int old_idx = i * d1 * d2 + j * d2 + k;
                        int new_idx;
                        if (dim0 == 1 && dim1 == 2) {
                            new_idx = i * d2 * d1 + k * d1 + j;
                        } else if (dim0 == 0 && dim1 == 2) {
                            new_idx = k * d1 * d0 + j * d0 + i;
                        } else if (dim0 == 0 && dim1 == 1) {
                            new_idx = j * d0 * d2 + i * d2 + k;
                        } else {
                            throw std::runtime_error("Unsupported transpose dims for 3D tensor");
                        }
                        new_data[new_idx] = data[old_idx];
                    }
                }
            }

        } else { // 4D
            int d0 = shape[0], d1 = shape[1], d2 = shape[2], d3 = shape[3];

            #pragma omp parallel for collapse(4)
            for (int i = 0; i < d0; ++i) {
                for (int j = 0; j < d1; ++j) {
                    for (int k = 0; k < d2; ++k) {
                        for (int l = 0; l < d3; ++l) {
                            int old_idx = i * d1 * d2 * d3 + j * d2 * d3 + k * d3 + l;
                            int new_idx;

                            if (dim0 == 2 && dim1 == 3) {
                                new_idx = i * d1 * d3 * d2 + j * d3 * d2 + l * d2 + k;
                            } else if (dim0 == 1 && dim1 == 3) {
                                new_idx = i * d3 * d2 * d1 + l * d2 * d1 + k * d1 + j;
                            } else if (dim0 == 1 && dim1 == 2) {
                                new_idx = i * d2 * d3 * d1 + k * d3 * d1 + l * d1 + j;
                            } else {
                                throw std::runtime_error("Unsupported transpose dimensions for 4D tensor");
                            }
                            new_data[new_idx] = data[old_idx];
                        }
                    }
                }
            }
        }

        return Tensor(new_data, new_shape);
    }
/*

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
    
    // Transpose dos ejes específicos
    Tensor transpose(int dim1, int dim2) const {
        std::vector<int> perm(shape.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[dim1], perm[dim2]);
        return transpose(perm);
    }
*/
    
    Tensor dropout_mask(float drop_prob, std::mt19937& rng) const {
        std::bernoulli_distribution dist(1.0 - drop_prob);
        std::vector<float> mask_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            mask_data[i] = dist(rng) ? 1.0f : 0.0f;
        return Tensor(mask_data, shape);
    }

    Tensor operator+(float scalar) const {
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result_data[i] = data[i] + scalar;
        return Tensor(result_data, shape);
    }

    Tensor operator-(float scalar) const {
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result_data[i] = data[i] - scalar;
        return Tensor(result_data, shape);
    }

    Tensor operator/(float scalar) const {
        if (scalar == 0.0f)
            throw std::runtime_error("Division by zero in Tensor");
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result_data[i] = data[i] / scalar;
        return Tensor(result_data, shape);
    }

    std::string shape_str() const {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            oss << shape[i];
            if (i != shape.size() - 1)
                oss << ", ";
        }
        oss << ")";
        return oss.str();
    }
    void clear() {
    data.clear();
    shape.clear();
}

};