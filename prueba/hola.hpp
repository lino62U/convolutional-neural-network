#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

// Forward declarations
class Optimizer;
class Loss;
class Metric;
class Layer;
class Activation;

// Enum for activation types
enum class ActivationType { ReLU, Sigmoid, Softmax , None};

// Tensor class
class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;
    Tensor(const std::vector<float>& d, const std::vector<int>& s) : data(d), shape(s) {
        if (d.size() != std::accumulate(s.begin(), s.end(), 1, std::multiplies<int>())) {
            throw std::runtime_error("Data size doesn't match shape");
        }
    }
    
    size_t size() const { return data.size(); }
    
    size_t total_elements() const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    Tensor matmul(const Tensor& other) const {

        //std::cout << "matmul shapes: {";
        //for (int s : shape) std::cout << s << ",";
        //std::cout << "} vs {";
        //for (int s : other.shape) std::cout << s << ",";
        //std::cout << "}\n";


        if (shape.size() < 2 || other.shape.size() < 2) {
            throw std::runtime_error("Matmul requires at least 2D tensors");
        }
        int batch_dim = shape.size() > 2 ? shape[0] : 1;
        int m = shape[shape.size() - 2];
        int k = shape[shape.size() - 1];
        int n = other.shape[other.shape.size() - 1];
        if (k != other.shape[other.shape.size() - 2]) {
            throw std::runtime_error("Invalid shapes for matmul: {" + std::to_string(m) + "," + std::to_string(k) +
                                    "} vs {" + std::to_string(other.shape[other.shape.size() - 2]) + "," + std::to_string(n) + "}");
        }

        std::vector<int> result_shape = shape;
        result_shape[result_shape.size() - 1] = n;
        std::vector<float> result_data(batch_dim * m * n, 0.0f);

        for (int b = 0; b < batch_dim; ++b) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l) {
                        int idx_a = shape.size() > 2 ? b * m * k + i * k + l : i * k + l;
                        int idx_b = other.shape.size() > 2 ? b * k * n + l * n + j : l * n + j;
                        sum += data[idx_a] * other.data[idx_b];
                    }
                    int idx_c = b * m * n + i * n + j;
                    result_data[idx_c] = sum;
                }
            }
        }
        return Tensor(result_data, result_shape);
    }

    Tensor operator+(const Tensor& other) const {
        // Check if shapes are compatible (equal or broadcasting possible)
        std::vector<int> result_shape = shape;
        if (other.shape.size() == 1 && shape.size() >= 2 && other.shape[0] == shape.back()) {
            // Broadcasting case: other is {dim}, this is {..., dim}
            std::vector<float> result_data(size(), 0.0f);
            int dim = shape.back();
            int outer_size = size() / dim;
            for (int i = 0; i < outer_size; ++i) {
                for (int d = 0; d < dim; ++d) {
                    result_data[i * dim + d] = data[i * dim + d] + other.data[d];
                }
            }
            return Tensor(result_data, shape);
        } else if (shape == other.shape) {
            // Standard addition
            std::vector<float> result_data(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                result_data[i] = data[i] + other.data[i];
            }
            return Tensor(result_data, shape);
        } else {
            throw std::runtime_error("Shape mismatch for addition: {" +
                                     std::to_string(shape[0]) + "," + std::to_string(shape[1]) + "," +
                                     std::to_string(shape[2]) + "} vs {" +
                                     std::to_string(other.shape.size() > 0 ? other.shape[0] : 0) + "," +
                                     std::to_string(other.shape.size() > 1 ? other.shape[1] : 0) + "," +
                                     std::to_string(other.shape.size() > 2 ? other.shape[2] : 0) + "}");
        }
    }

    Tensor dropout_mask(float dropout_rate, std::mt19937& rng) const {
        if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
            throw std::runtime_error("Invalid dropout rate");
        }
        std::vector<float> mask_data(data.size());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float scale = 1.0f / (1.0f - dropout_rate); // Scale to maintain expected value
        for (size_t i = 0; i < data.size(); ++i) {
            mask_data[i] = (dist(rng) >= dropout_rate) ? scale : 0.0f;
        }
        return Tensor(mask_data, shape); // Same shape as input tensor
    }

    Tensor operator*(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::runtime_error("Shape mismatch for element-wise multiplication");
        }
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result_data[i] = data[i] * other.data[i];
        }
        return Tensor(result_data, shape);
    }
    Tensor operator/(float scalar) const {
        if (scalar == 0.0f) {
            throw std::runtime_error("Division by zero in Tensor operator/");
        }
        std::vector<float> result_data(size());
        for (size_t i = 0; i < size(); ++i) {
            result_data[i] = data[i] / scalar;
        }
        return Tensor(result_data, shape);
    }

    
    Tensor transpose(int dim0 = -1, int dim1 = -1) const {
    if (shape.size() < 2 || shape.size() > 4) {
        throw std::runtime_error("Transpose supported for 2D, 3D, or 4D tensors only");
    }

    if (dim0 == -1 && dim1 == -1) {
        // Default: transpose last two dimensions
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
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                new_data[j * rows + i] = data[i * cols + j];
            }
        }
    } else if (shape.size() == 3) {
        int d0 = shape[0];
        int d1 = shape[1];
        int d2 = shape[2];
        for (int i = 0; i < d0; ++i) {
            for (int j = 0; j < d1; ++j) {
                for (int k = 0; k < d2; ++k) {
                    int old_idx = i * d1 * d2 + j * d2 + k;
                    int new_idx;
                    if (dim0 == 1 && dim1 == 2) {
                        new_idx = i * d2 * d1 + k * d1 + j;
                    } else if (dim0 == 0 && dim1 == 2) {
                        new_idx = k * d1 * d0 + j * d0 + i;
                    } else { // dim0 == 0, dim1 == 1
                        new_idx = j * d0 * d2 + i * d2 + k;
                    }
                    new_data[new_idx] = data[old_idx];
                }
            }
        }
    } else { // 4D
        int d0 = shape[0];
        int d1 = shape[1];
        int d2 = shape[2];
        int d3 = shape[3];
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
                        } else { // Add other cases as needed
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

    Tensor softmax() const {
    if (shape.size() != 2 && shape.size() != 4) {
        throw std::runtime_error("Softmax expects 2D or 4D input tensor");
    }

    std::vector<float> result_data(data.size());
    if (shape.size() == 2) {
        // 2D case: Apply softmax across last dimension
        int rows = shape[0];
        int cols = shape[1];
        for (int i = 0; i < rows; ++i) {
            float max_val = *std::max_element(data.begin() + i * cols, data.begin() + (i + 1) * cols);
            float sum = 0.0f;
            for (int j = 0; j < cols; ++j) {
                int idx = i * cols + j;
                result_data[idx] = std::exp(data[idx] - max_val);
                sum += result_data[idx];
            }
            for (int j = 0; j < cols; ++j) {
                result_data[i * cols + j] /= sum;
            }
        }
        return Tensor(result_data, shape);
    } else {
        // 4D case: Apply softmax across last dimension
        int batch_size = shape[0];
        int num_heads = shape[1];
        int seq_len = shape[2];
        int softmax_dim = shape[3];
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    // Compute max for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < softmax_dim; ++j) {
                        int idx = n * num_heads * seq_len * softmax_dim +
                                 h * seq_len * softmax_dim + i * softmax_dim + j;
                        max_val = std::max(max_val, data[idx]);
                    }
                    // Compute exp and sum
                    float sum = 0.0f;
                    for (int j = 0; j < softmax_dim; ++j) {
                        int idx = n * num_heads * seq_len * softmax_dim +
                                 h * seq_len * softmax_dim + i * softmax_dim + j;
                        result_data[idx] = std::exp(data[idx] - max_val);
                        sum += result_data[idx];
                    }
                    // Normalize
                    for (int j = 0; j < softmax_dim; ++j) {
                        int idx = n * num_heads * seq_len * softmax_dim +
                                 h * seq_len * softmax_dim + i * softmax_dim + j;
                        result_data[idx] /= sum;
                    }
                }
            }
        }
        return Tensor(result_data, shape);
    }
}

Tensor sum(int axis) const {
        if (axis < 0 || axis >= static_cast<int>(shape.size())) {
            throw std::runtime_error("Invalid axis for sum");
        }
        // Compute output shape by removing the specified axis
        std::vector<int> out_shape = shape;
        out_shape.erase(out_shape.begin() + axis);
        if (out_shape.empty()) {
            out_shape.push_back(1); // Scalar result
        }
        std::vector<float> out_data(std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>()), 0.0f);

        // Compute strides for input tensor
        std::vector<int> strides(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Sum along the specified axis
        for (size_t i = 0; i < data.size(); ++i) {
            // Compute output index by excluding the axis dimension
            std::vector<int> indices(shape.size());
            int temp = i;
            for (int j = 0; j < static_cast<int>(shape.size()); ++j) {
                indices[j] = temp / strides[j];
                temp %= strides[j];
            }
            std::vector<int> out_indices;
            for (int j = 0; j < static_cast<int>(shape.size()); ++j) {
                if (j != axis) {
                    out_indices.push_back(indices[j]);
                }
            }
            int out_idx = 0;
            for (size_t j = 0; j < out_indices.size(); ++j) {
                int stride = 1;
                for (size_t k = j + 1; k < out_shape.size(); ++k) {
                    stride *= out_shape[k];
                }
                out_idx += out_indices[j] * stride;
            }
            out_data[out_idx] += data[i];
        }

        return Tensor(out_data, out_shape);
    }
static Tensor matmul_transpose(const Tensor& A, const Tensor& B, bool transpose_A = false, bool transpose_B = false) {
    Tensor A_eff = transpose_A ? A.transpose() : A;
    Tensor B_eff = transpose_B ? B.transpose() : B;

    //std::cout << "matmul_transpose shapes: {";
    //for (auto d : A_eff.shape) std::cout << d << ",";
    //std::cout << "} vs {";
    //for (auto d : B_eff.shape) std::cout << d << ",";
    //std::cout << "}\n";

    return A_eff.matmul(B_eff);
}
std::string shape_string() const {
    std::string s = "{";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i < shape.size() - 1) s += ",";
    }
    s += "}";
    return s;
}


Tensor reshape(const std::vector<int>& new_shape) const {
    int new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_total != this->size()) {
        throw std::runtime_error("Reshape error: total elements mismatch.");
    }
    return Tensor(this->data, new_shape);
}


Tensor operator-(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch for subtraction");
    }
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = data[i] - other.data[i];
    }
    return Tensor(result_data, shape);
}
Tensor operator-(float scalar) const {
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = data[i] - scalar;
    }
    return Tensor(result_data, shape);
}






};

// MNIST Loader class
class MNISTLoader {
public:
    Tensor images;
    Tensor labels;

    MNISTLoader(const std::string& image_file, const std::string& label_file, int max_samples = -1) {
        load_images(image_file, max_samples);
        load_labels(label_file, max_samples);
    }

private:
    int read_int32(std::ifstream& file) {
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    }

    void load_images(const std::string& filename, int max_samples) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open image file: " + filename);
        }

        int magic = read_int32(file);
        int num_images = read_int32(file);
        int rows = read_int32(file);
        int cols = read_int32(file);

        if (magic != 2051) {
            throw std::runtime_error("Invalid MNIST image file format");
        }

        if (max_samples > 0 && max_samples < num_images) {
            num_images = max_samples;
        }

        std::vector<float> data(num_images * rows * cols);
        for (int i = 0; i < num_images * rows * cols; ++i) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            data[i] = pixel / 255.0f;
        }
        file.close();

        images = Tensor(data, {num_images, 1, rows, cols});
    }

    void load_labels(const std::string& filename, int max_samples) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open label file: " + filename);
        }

        int magic = read_int32(file);
        int num_labels = read_int32(file);

        if (magic != 2049) {
            throw std::runtime_error("Invalid MNIST label file format");
        }

        if (max_samples > 0 && max_samples < num_labels) {
            num_labels = max_samples;
        }

        std::vector<float> data(num_labels * 10, 0.0f);
        for (int i = 0; i < num_labels; ++i) {
            unsigned char label;
            file.read(reinterpret_cast<char*>(&label), 1);
            data[i * 10 + label] = 1.0f;
        }
        file.close();

        labels = Tensor(data, {num_labels, 10});
    }
};

// Activation function class
class Activation {
public:
    virtual float apply(float x) const = 0;
    virtual float derivative(float x) const = 0;
    virtual Tensor apply_batch(const Tensor& input) const {
        std::vector<float> output(input.data.size());
        for (size_t i = 0; i < input.data.size(); ++i) {
            output[i] = apply(input.data[i]);
        }
        return Tensor(output, input.shape);
    }
    virtual Tensor derivative_batch(const Tensor& input, const Tensor& output) const {
        std::vector<float> grad(input.data.size());
        for (size_t i = 0; i < input.data.size(); ++i) {
            grad[i] = derivative(input.data[i]);
        }
        return Tensor(grad, input.shape);
    }
    virtual ~Activation() {}
};

class ReLU : public Activation {
public:
    float apply(float x) const override { return std::max(0.0f, x); }
    float derivative(float x) const override { return x > 0 ? 1.0f : 0.0f; }
};

class Sigmoid : public Activation {
public:
    float apply(float x) const override { return 1.0f / (1.0f + std::exp(-x)); }
    float derivative(float x) const override {
        float s = apply(x);
        return s * (1.0f - s);
    }
};

class Softmax : public Activation {
public:
    float apply(float x) const override { return x; }
    float derivative(float x) const override { return 1.0f; }

    Tensor apply_batch(const Tensor& input) const override {
        return input.softmax();
    }

    Tensor derivative_batch(const Tensor& input, const Tensor& output) const override {
        return Tensor(std::vector<float>(input.data.size(), 1.0f), input.shape);
    }
};

// Optimizer base class
class Optimizer {
public:
    virtual void update(Tensor& param, const Tensor& grad) = 0;
    virtual ~Optimizer() {}
};

class Adam : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;

    struct Moments {
        Tensor m;
        Tensor v;
    };

    std::unordered_map<Tensor*, Moments> moments_map;

public:
    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void update(Tensor& param, const Tensor& grad) override {
        if (param.total_elements() != grad.total_elements()) {
            std::cerr << "❌ param.shape = " << param.shape_string()
                    << ", grad.shape = " << grad.shape_string() << "\n";
            throw std::runtime_error("Size mismatch in optimizer update");
        }

        Moments& moments = moments_map[&param];

        if (moments.m.data.empty()) {
            moments.m = Tensor(std::vector<float>(param.size(), 0.0f), param.shape);
            moments.v = Tensor(std::vector<float>(param.size(), 0.0f), param.shape);
        }

        ++t;

        for (size_t i = 0; i < param.data.size(); ++i) {
            moments.m.data[i] = beta1 * moments.m.data[i] + (1.0f - beta1) * grad.data[i];
            moments.v.data[i] = beta2 * moments.v.data[i] + (1.0f - beta2) * grad.data[i] * grad.data[i];
        }

        float beta1_t = std::pow(beta1, t);
        float beta2_t = std::pow(beta2, t);

        for (size_t i = 0; i < param.data.size(); ++i) {
            float m_hat = moments.m.data[i] / (1.0f - beta1_t);
            float v_hat = moments.v.data[i] / (1.0f - beta2_t);
            param.data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};

// Layer base class
class Layer {
public:
    virtual Tensor forward(const Tensor& input, bool training = false) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update_weights(Optimizer* optimizer) = 0;
    virtual size_t num_params() const = 0;
    virtual ~Layer() {}
};

class GlobalAveragePooling : public Layer {
private:
    Tensor input_cache;

public:
    GlobalAveragePooling() {}

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3) {
            throw std::runtime_error("GlobalAveragePooling expects 3D input tensor");
        }
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int embed_dim = input.shape[2];

        std::vector<float> output_data(batch_size * embed_dim, 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int d = 0; d < embed_dim; ++d) {
                float sum = 0.0f;
                for (int s = 0; s < seq_len; ++s) {
                    sum += input.data[n * seq_len * embed_dim + s * embed_dim + d];
                }
                output_data[n * embed_dim + d] = sum / seq_len;
            }
        }
        input_cache = input;
        return Tensor(output_data, std::vector<int>{batch_size, embed_dim});
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape.size() != 2 || grad_output.shape[0] != input_cache.shape[0] ||
            grad_output.shape[1] != input_cache.shape[2]) {
            throw std::runtime_error("Gradient shape mismatch in GlobalAveragePooling backward");
        }
        int batch_size = input_cache.shape[0];
        int seq_len = input_cache.shape[1];
        int embed_dim = input_cache.shape[2];

        std::vector<float> grad_input_data(batch_size * seq_len * embed_dim, 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < embed_dim; ++d) {
                    grad_input_data[n * seq_len * embed_dim + s * embed_dim + d] =
                        grad_output.data[n * embed_dim + d] / seq_len;
                }
            }
        }
        return Tensor(grad_input_data, std::vector<int>{batch_size, seq_len, embed_dim});
    }

    void update_weights(Optimizer* optimizer) override {}
    size_t num_params() const override { return 0; }
};

class Dropout : public Layer {
private:
    float dropout_rate;
    Tensor mask;
    std::mt19937 rng;

public:
    Dropout(float rate) : dropout_rate(rate), rng(std::random_device{}()) {
        if (rate < 0.0f || rate >= 1.0f) {
            throw std::runtime_error("Dropout rate must be in [0, 1)");
        }
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (!training) {
            std::vector<float> output(input.data.size());
            for (size_t i = 0; i < input.data.size(); ++i) {
                output[i] = input.data[i] * (1.0f - dropout_rate);
            }
            return Tensor(output, input.shape);
        }

        std::vector<float> mask_data(input.data.size());
        std::vector<float> output(input.data.size());
        std::bernoulli_distribution dist(1.0f - dropout_rate);
        for (size_t i = 0; i < input.data.size(); ++i) {
            mask_data[i] = dist(rng) ? 1.0f / (1.0f - dropout_rate) : 0.0f;
            output[i] = input.data[i] * mask_data[i];
        }
        mask = Tensor(mask_data, input.shape);
        return Tensor(output, input.shape);
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape != mask.shape) {
            throw std::runtime_error("Shape mismatch in dropout backward");
        }
        std::vector<float> grad_input(grad_output.data.size());
        for (size_t i = 0; i < grad_output.data.size(); ++i) {
            grad_input[i] = grad_output.data[i] * mask.data[i];
        }
        return Tensor(grad_input, grad_output.shape);
    }

    void update_weights(Optimizer* optimizer) override {}
    size_t num_params() const override { return 0; }
};


class LayerNorm : public Layer {
private:
    float epsilon;
    Tensor gamma, beta;
    Tensor gamma_grad, beta_grad; // Added gradient tensors
    Tensor input_cache;

public:
    LayerNorm(int dim, float eps = 1e-5f) : epsilon(eps) {
        std::vector<float> gamma_data(dim, 1.0f);
        std::vector<float> beta_data(dim, 0.0f);
        std::vector<float> grad_data(dim, 0.0f); // Initialize gradients to zero
        gamma = Tensor(gamma_data, std::vector<int>{dim});
        beta = Tensor(beta_data, std::vector<int>{dim});
        gamma_grad = Tensor(grad_data, std::vector<int>{dim});
        beta_grad = Tensor(grad_data, std::vector<int>{dim});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3) {
            throw std::runtime_error("LayerNorm expects 3D input tensor");
        }
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int dim = input.shape[2];
        if (dim != gamma.shape[0]) {
            throw std::runtime_error("LayerNorm dimension mismatch");
        }

        std::vector<float> output_data(input.data.size());
        for (int n = 0; n < batch_size; ++n) {
            for (int s = 0; s < seq_len; ++s) {
                // Compute mean
                float mean = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    mean += input.data[n * seq_len * dim + s * dim + d];
                }
                mean /= dim;

                // Compute variance
                float var = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = input.data[n * seq_len * dim + s * dim + d] - mean;
                    var += diff * diff;
                }
                var /= dim;
                var = std::sqrt(var + epsilon);

                // Normalize and scale
                for (int d = 0; d < dim; ++d) {
                    int idx = n * seq_len * dim + s * dim + d;
                    output_data[idx] = (input.data[idx] - mean) / var * gamma.data[d] + beta.data[d];
                }
            }
        }
        input_cache = input;
        return Tensor(output_data, input.shape);
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape != input_cache.shape) {
            throw std::runtime_error("Gradient shape mismatch in LayerNorm backward");
        }
        int batch_size = input_cache.shape[0];
        int seq_len = input_cache.shape[1];
        int dim = input_cache.shape[2];

        std::vector<float> grad_input_data(input_cache.data.size(), 0.0f);
        std::vector<float> grad_gamma_data(dim, 0.0f);
        std::vector<float> grad_beta_data(dim, 0.0f);

        for (int n = 0; n < batch_size; ++n) {
            for (int s = 0; s < seq_len; ++s) {
                // Recompute mean and variance
                float mean = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    mean += input_cache.data[n * seq_len * dim + s * dim + d];
                }
                mean /= dim;

                float var = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = input_cache.data[n * seq_len * dim + s * dim + d] - mean;
                    var += diff * diff;
                }
                var /= dim;
                var = std::sqrt(var + epsilon);

                // Compute normalized input
                std::vector<float> x_hat(dim);
                for (int d = 0; d < dim; ++d) {
                    x_hat[d] = (input_cache.data[n * seq_len * dim + s * dim + d] - mean) / var;
                }

                // Gradients for gamma and beta
                for (int d = 0; d < dim; ++d) {
                    grad_gamma_data[d] += grad_output.data[n * seq_len * dim + s * dim + d] * x_hat[d];
                    grad_beta_data[d] += grad_output.data[n * seq_len * dim + s * dim + d];
                }

                // Gradient for input
                float sum_grad = 0.0f;
                float sum_grad_x_hat = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    sum_grad += grad_output.data[n * seq_len * dim + s * dim + d] * gamma.data[d];
                    sum_grad_x_hat += grad_output.data[n * seq_len * dim + s * dim + d] * gamma.data[d] * x_hat[d];
                }
                for (int d = 0; d < dim; ++d) {
                    int idx = n * seq_len * dim + s * dim + d;
                    grad_input_data[idx] = (grad_output.data[idx] * gamma.data[d] - sum_grad / dim -
                                           x_hat[d] * sum_grad_x_hat / dim) / var;
                }
            }
        }

        gamma_grad = Tensor(grad_gamma_data, std::vector<int>{dim});
        beta_grad = Tensor(grad_beta_data, std::vector<int>{dim});
        return Tensor(grad_input_data, input_cache.shape);
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(gamma, gamma_grad);
        optimizer->update(beta, beta_grad);
    }

    size_t num_params() const override {
        return gamma.data.size() + beta.data.size();
    }
};
/*
class LayerNorm : public Layer {
private:
    float epsilon;
    Tensor gamma, beta;
    Tensor gamma_grad, beta_grad;

    // Cache
    Tensor input_cache;
    Tensor mean_cache, var_cache, x_hat_cache;

public:
    LayerNorm(int dim, float eps = 1e-5f) : epsilon(eps) {
        gamma = Tensor(std::vector<float>(dim, 1.0f), {dim});
        beta  = Tensor(std::vector<float>(dim, 0.0f), {dim});
        gamma_grad = Tensor(std::vector<float>(dim, 0.0f), {dim});
        beta_grad  = Tensor(std::vector<float>(dim, 0.0f), {dim});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3)
            throw std::runtime_error("LayerNorm expects 3D input");

        int B = input.shape[0], T = input.shape[1], D = input.shape[2];
        input_cache = input;

        std::vector<float> output_data(B * T * D);
        std::vector<float> mean_data(B * T), var_data(B * T);
        std::vector<float> x_hat_data(B * T * D);

        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                int offset = (b * T + t) * D;

                // Mean
                float mean = 0.0f;
                for (int d = 0; d < D; ++d)
                    mean += input.data[offset + d];
                mean /= D;

                // Variance (sin sqrt)
                float var = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float diff = input.data[offset + d] - mean;
                    var += diff * diff;
                }
                var /= D;

                float std = std::sqrt(var + epsilon);
                mean_data[b * T + t] = mean;
                var_data[b * T + t]  = var;

                // Normalize + Scale + Shift
                for (int d = 0; d < D; ++d) {
                    float x_hat = (input.data[offset + d] - mean) / std;
                    x_hat_data[offset + d] = x_hat;
                    output_data[offset + d] = x_hat * gamma.data[d] + beta.data[d];
                }
            }
        }

        mean_cache = Tensor(mean_data, {B, T});
        var_cache  = Tensor(var_data, {B, T});
        x_hat_cache = Tensor(x_hat_data, input.shape);

        return Tensor(output_data, input.shape);
    }

    Tensor backward(const Tensor& grad_output) override {
        const auto& x = input_cache;
        int B = x.shape[0], T = x.shape[1], D = x.shape[2];

        std::vector<float> grad_input_data(B * T * D, 0.0f);
        std::vector<float> grad_gamma_data(D, 0.0f);
        std::vector<float> grad_beta_data(D, 0.0f);

        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                int offset = (b * T + t) * D;
                float var = var_cache.data[b * T + t];
                float std = std::sqrt(var + epsilon);

                // dL/dgamma y dL/dbeta
                for (int d = 0; d < D; ++d) {
                    float grad_out = grad_output.data[offset + d];
                    float x_hat    = x_hat_cache.data[offset + d];
                    grad_gamma_data[d] += grad_out * x_hat;
                    grad_beta_data[d]  += grad_out;
                }

                // dL/dx (derivada completa)
                float sum1 = 0.0f, sum2 = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float dx_hat = grad_output.data[offset + d] * gamma.data[d];
                    sum1 += dx_hat;
                    sum2 += dx_hat * x_hat_cache.data[offset + d];
                }

                for (int d = 0; d < D; ++d) {
                    float dx_hat = grad_output.data[offset + d] * gamma.data[d];
                    float x_hat  = x_hat_cache.data[offset + d];
                    grad_input_data[offset + d] =
                        (dx_hat - sum1 / D - x_hat * sum2 / D) / std;
                }
            }
        }

        gamma_grad = Tensor(grad_gamma_data, {D});
        beta_grad  = Tensor(grad_beta_data, {D});
        return Tensor(grad_input_data, x.shape);
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(gamma, gamma_grad);
        optimizer->update(beta, beta_grad);
    }

    size_t num_params() const override {
        return gamma.total_elements() + beta.total_elements();
    }
};

*/

class PatchEmbedding : public Layer {
private:
    int patch_size;
    int in_channels;
    int embed_dim;
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    Tensor z_cache;
    std::mt19937 rng;

    void initialize_weights(int patch_dim, int embed_dim) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / patch_dim));
        std::vector<float> w_data(patch_dim * embed_dim);
        for (float& w : w_data) {
            w = dist(rng);
        }
        weights = Tensor(w_data, {patch_dim, embed_dim});
        std::vector<float> b_data(embed_dim, 0.0f);
        bias = Tensor(b_data, {embed_dim});
    }

public:
    PatchEmbedding(int in_channels, int patch_size, int embed_dim)
        : in_channels(in_channels), patch_size(patch_size), embed_dim(embed_dim), rng(std::random_device{}()) {
        if (patch_size <= 0 || in_channels <= 0 || embed_dim <= 0) {
            throw std::runtime_error("Invalid patch embedding parameters");
        }
        int patch_dim = in_channels * patch_size * patch_size;
        initialize_weights(patch_dim, embed_dim);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        //std::cout << "📐 Weights shape (forward): {" << weights.shape[0] << "," << weights.shape[1] << "}\n";

        if (input.shape.size() != 4 || input.shape[1] != in_channels) {
            throw std::runtime_error("Invalid input shape for patch embedding");
        }
        int batch_size = input.shape[0];
        int height = input.shape[2];
        int width = input.shape[3];
        if (height % patch_size != 0 || width % patch_size != 0) {
            throw std::runtime_error("Image dimensions must be divisible by patch size");
        }

        //std::cout << "PatchEmbedding input shape: {" << input.shape[0] << "," << input.shape[1] << "," << input.shape[2] << "," << input.shape[3] << "}\n";


        int num_patches = (height / patch_size) * (width / patch_size);
        int patch_dim = in_channels * patch_size * patch_size;

        std::vector<float> patches(batch_size * num_patches * patch_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < height; h += patch_size) {
                for (int w = 0; w < width; w += patch_size) {
                    int patch_idx = (h / patch_size) * (width / patch_size) + (w / patch_size);
                    for (int c = 0; c < in_channels; ++c) {
                        for (int ph = 0; ph < patch_size; ++ph) {
                            for (int pw = 0; pw < patch_size; ++pw) {
                                int idx = n * num_patches * patch_dim + patch_idx * patch_dim +
                                         c * patch_size * patch_size + ph * patch_size + pw;
                                patches[idx] = input.data[n * in_channels * height * width +
                                                         c * height * width + (h + ph) * width + (w + pw)];
                            }
                        }
                    }
                }
            }
        }

        Tensor patch_tensor(patches, {batch_size * num_patches, patch_dim});
        input_cache = patch_tensor;
        z_cache = patch_tensor.matmul(weights) + bias;
        return Tensor(z_cache.data, {batch_size, num_patches, embed_dim});
    }
    Tensor backward(const Tensor& grad_output) {
        //std::cout << "grad_output shape: {" << grad_output.shape[0] << "," << grad_output.shape[1] << "," << grad_output.shape[2] << "}\n";
        //std::cout << "input_cache shape: {" << input_cache.shape[0] << "," << input_cache.shape[1] << "}\n";

        // Verificación correcta
        int batch_size = grad_output.shape[0];
        int num_patches = grad_output.shape[1];
        int embed_dim = grad_output.shape[2];
        int patch_dim = input_cache.shape[1];

        if (grad_output.shape.size() != 3 ||
            batch_size * num_patches != input_cache.shape[0] ||
            embed_dim != weights.shape[1]) {
            std::cerr << "grad_output shape: {" 
                    << grad_output.shape[0] << "," 
                    << grad_output.shape[1] << "," 
                    << grad_output.shape[2] << "} vs input_cache shape: {" 
                    << input_cache.shape[0] << "," 
                    << input_cache.shape[1] << "}\n";
            throw std::runtime_error("Gradient shape mismatch in patch embedding backward");
        }

        // Reorganizar grad_output a forma (batch_size * num_patches, embed_dim)
        std::vector<float> flat_grad_z(batch_size * num_patches * embed_dim);
        for (int b = 0; b < batch_size; ++b) {
            for (int p = 0; p < num_patches; ++p) {
                for (int e = 0; e < embed_dim; ++e) {
                    int src_idx = b * num_patches * embed_dim + p * embed_dim + e;
                    int dst_idx = (b * num_patches + p) * embed_dim + e;
                    flat_grad_z[dst_idx] = grad_output.data[src_idx];
                }
            }
        }
        Tensor grad_z(flat_grad_z, {batch_size * num_patches, embed_dim});

        
        // Gradientes con respecto a pesos y bias
        Tensor grad_weights = Tensor::matmul_transpose(input_cache, grad_z, true, false);



        std::vector<float> grad_bias_data(embed_dim, 0.0f);
        for (int i = 0; i < batch_size * num_patches; ++i) {
            for (int j = 0; j < embed_dim; ++j) {
                grad_bias_data[j] += grad_z.data[i * embed_dim + j];
            }
        }
        Tensor grad_bias(grad_bias_data, {embed_dim});

    //std::cout << "📐 input_cache shape: {" << input_cache.shape[0] << "," << input_cache.shape[1] << "}\n";
    //std::cout << "📐 weights shape (backward): {" << weights.shape[0] << "," << weights.shape[1] << "}\n";


        // Gradiente con respecto a las entradas planas (patches)
        Tensor grad_patches = grad_z.matmul(weights.transpose());

        // Reconstrucción a imagen
        int height = static_cast<int>(std::sqrt(num_patches)) * patch_size;
        int width = height;

        std::vector<float> grad_input_data(batch_size * in_channels * height * width, 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < num_patches; ++p) {
                int h = (p / (width / patch_size)) * patch_size;
                int w = (p % (width / patch_size)) * patch_size;
                for (int c = 0; c < in_channels; ++c) {
                    for (int ph = 0; ph < patch_size; ++ph) {
                        for (int pw = 0; pw < patch_size; ++pw) {
                            int img_idx = n * in_channels * height * width + c * height * width + (h + ph) * width + (w + pw);
                            int patch_idx = n * num_patches * patch_dim + p * patch_dim + c * patch_size * patch_size + ph * patch_size + pw;
                            grad_input_data[img_idx] = grad_patches.data[patch_idx];
                        }
                    }
                }
            }
        }

        if (weights_grad.shape != weights.shape) {
            weights_grad = Tensor(std::vector<float>(weights.total_elements(), 0.0f), weights.shape);
        }

        if (bias_grad.shape != bias.shape || bias_grad.data.size() != bias.data.size()) {
    bias_grad = Tensor(std::vector<float>(bias.total_elements(), 0.0f), bias.shape);
}
        std::copy(grad_weights.data.begin(), grad_weights.data.end(), weights_grad.data.begin());
        std::copy(grad_bias.data.begin(), grad_bias.data.end(), bias_grad.data.begin());

        //std::cout << "grad_bias shape: {" << grad_bias.shape[0] << "} size: " << grad_bias.data.size() << "\n";


        return Tensor(grad_input_data, {batch_size, in_channels, height, width});
    }


    void update_weights(Optimizer* optimizer) override {
        /*
        std::cout << "⏬ Updating weights...\n";
        std::cout << "weights shape: ";
        for (int s : weights.shape) std::cout << s << ",";
        std::cout << "\nweights_grad shape: ";
        for (int s : weights_grad.shape) std::cout << s << ",";
        std::cout << "\n";

        std::cout << "bias shape: ";
        for (int s : bias.shape) std::cout << s << ",";
        std::cout << "\nbias_grad shape: ";
        for (int s : bias_grad.shape) std::cout << s << ",";
        std::cout << "\n";
        */
        optimizer->update(weights, weights_grad);
        optimizer->update(bias, bias_grad);
    }

    size_t num_params() const override {
        return weights.total_elements() + bias.total_elements();
    }

private:
    Tensor weights_grad;
    Tensor bias_grad;
};

class PositionalEncoding : public Layer {
private:
    Tensor encodings;
    bool learnable;
    Tensor encodings_grad;

public:
    PositionalEncoding(int num_patches, int embed_dim, bool learnable = false)
        : learnable(learnable) {
        std::vector<float> enc_data(num_patches * embed_dim);
        if (!learnable) {
            for (int pos = 0; pos < num_patches; ++pos) {
                for (int i = 0; i < embed_dim; ++i) {
                    if (i % 2 == 0) {
                        enc_data[pos * embed_dim + i] = std::sin(pos / std::pow(10000.0f, i / (float)embed_dim));
                    } else {
                        enc_data[pos * embed_dim + i] = std::cos(pos / std::pow(10000.0f, (i - 1) / (float)embed_dim));
                    }
                }
            }
        } else {
            std::mt19937 rng(std::random_device{}());
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (float& e : enc_data) {
                e = dist(rng);
            }
        }
        encodings = Tensor(enc_data, {num_patches, embed_dim});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape[1] != encodings.shape[0] || input.shape[2] != encodings.shape[1]) {
            throw std::runtime_error("Shape mismatch in positional encoding");
        }
        std::vector<float> output_data(input.data.size());
        for (int n = 0; n < input.shape[0]; ++n) {
            for (int p = 0; p < input.shape[1]; ++p) {
                for (int d = 0; d < input.shape[2]; ++d) {
                    output_data[n * input.shape[1] * input.shape[2] + p * input.shape[2] + d] =
                        input.data[n * input.shape[1] * input.shape[2] + p * input.shape[2] + d] +
                        encodings.data[p * input.shape[2] + d];
                }
            }
        }
        return Tensor(output_data, input.shape);
    }

    Tensor backward(const Tensor& grad_output) override {
        if (!learnable) {
            return grad_output;
        }
        std::vector<float> grad_enc_data(encodings.size(), 0.0f);
        for (int n = 0; n < grad_output.shape[0]; ++n) {
            for (int p = 0; p < grad_output.shape[1]; ++p) {
                for (int d = 0; d < grad_output.shape[2]; ++d) {
                    grad_enc_data[p * grad_output.shape[2] + d] +=
                        grad_output.data[n * grad_output.shape[1] * grad_output.shape[2] + p * grad_output.shape[2] + d];
                }
            }
        }
        encodings_grad = Tensor(grad_enc_data, encodings.shape);
        return grad_output;
    }

    void update_weights(Optimizer* optimizer) override {
        if (learnable) {
            optimizer->update(encodings, encodings_grad);
        }
    }

    size_t num_params() const override {
        return learnable ? encodings.total_elements() : 0;
    }
};

class MultiHeadSelfAttention : public Layer {
private:
    int embed_dim;
    int num_heads;
    Tensor W_q, W_k, W_v, W_o;
    Tensor bias_q, bias_k, bias_v, bias_o;
    Tensor q_cache, k_cache, v_cache, attn_cache, input_cache, out_2d_final_cache;
    float dropout_rate;
    Tensor mask;
    std::mt19937 rng;
    Tensor W_q_grad, W_k_grad, W_v_grad, W_o_grad;
    Tensor bias_q_grad, bias_k_grad, bias_v_grad, bias_o_grad;
    
    void initialize_weights() {
        int head_dim = embed_dim / num_heads;
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim));
        std::vector<float> w_data(embed_dim * embed_dim);
        std::vector<float> b_data(embed_dim);
        for (float& w : w_data) w = dist(rng);
        for (float& b : b_data) b = 0.0f;
        W_q = Tensor(w_data, {embed_dim, embed_dim});
        W_k = Tensor(w_data, {embed_dim, embed_dim});
        W_v = Tensor(w_data, {embed_dim, embed_dim});
        W_o = Tensor(w_data, {embed_dim, embed_dim});
        bias_q = Tensor(b_data, {embed_dim});
        bias_k = Tensor(b_data, {embed_dim});
        bias_v = Tensor(b_data, {embed_dim});
        bias_o = Tensor(b_data, {embed_dim});
    }
    /*
    void initialize_weights() {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim));

        auto init_weight = [&](Tensor& W) {
            std::vector<float> data(embed_dim * embed_dim);
            for (float& w : data) w = dist(rng);
            W = Tensor(data, {embed_dim, embed_dim});
        };

        auto init_bias = [&](Tensor& b) {
            std::vector<float> data(embed_dim, 0.0f);
            b = Tensor(data, {embed_dim});
        };

        init_weight(W_q);
        init_weight(W_k);
        init_weight(W_v);
        init_weight(W_o);

        init_bias(bias_q);
        init_bias(bias_k);
        init_bias(bias_v);
        init_bias(bias_o);
    }
    */

public:
    MultiHeadSelfAttention(int embed_dim, int num_heads, float dropout_rate = 0.0f)
        : embed_dim(embed_dim), num_heads(num_heads), dropout_rate(dropout_rate), rng(std::random_device{}()) {
        if (embed_dim % num_heads != 0) {
            throw std::runtime_error("embed_dim must be divisible by num_heads");
        }
        initialize_weights();
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int head_dim = embed_dim / num_heads;
        input_cache = input;

        // Reshape input to 2D for matrix multiplication
        Tensor input_2d(input.data, {batch_size * seq_len, embed_dim});
        Tensor Q = input_2d.matmul(W_q) + bias_q;
        Tensor K = input_2d.matmul(W_k) + bias_k;
        Tensor V = input_2d.matmul(W_v) + bias_v;

        // Reshape Q, K, V to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> Q_new(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> K_new(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> V_new(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        Q_new[idx] = Q.data[src_idx];
                        K_new[idx] = K.data[src_idx];
                        V_new[idx] = V.data[src_idx];
                    }
                }
            }
        }
        Q = Tensor(Q_new, {batch_size, seq_len, num_heads, head_dim});
        K = Tensor(K_new, {batch_size, seq_len, num_heads, head_dim});
        V = Tensor(V_new, {batch_size, seq_len, num_heads, head_dim});
        q_cache = Q;
        k_cache = K;
        v_cache = V;

        // Reshape Q, K to [batch_size * num_heads, seq_len, head_dim]
        std::vector<float> Q_attn(batch_size * num_heads * seq_len * head_dim);
        std::vector<float> K_attn(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        Q_attn[idx] = Q.data[src_idx];
                        K_attn[idx] = K.data[src_idx];
                    }
                }
            }
        }
        Tensor Q_2d(Q_attn, {batch_size * num_heads, seq_len, head_dim});
        Tensor K_2d(K_attn, {batch_size * num_heads, seq_len, head_dim});

        // Transpose K to [batch_size * num_heads, head_dim, seq_len]
        std::vector<float> K_trans_data(batch_size * num_heads * head_dim * seq_len);
        for (int n = 0; n < batch_size * num_heads; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int d = 0; d < head_dim; ++d) {
                    K_trans_data[n * head_dim * seq_len + d * seq_len + p] = K_2d.data[n * seq_len * head_dim + p * head_dim + d];
                }
            }
        }
        Tensor K_trans(K_trans_data, {batch_size * num_heads, head_dim, seq_len});

        // Compute scores
        Tensor scores_2d = Q_2d.matmul(K_trans) / std::sqrt((float)head_dim);

        //std::cout << "MHSA scores_2d shape: {";
        //for (int s : scores_2d.shape) std::cout << s << ",";
        //std::cout << "}\n";

        // Reshape scores to [batch_size, num_heads, seq_len, seq_len]
        std::vector<float> scores_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        int src_idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                        scores_data[idx] = scores_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor scores(scores_data, {batch_size, num_heads, seq_len, seq_len});

        // Softmax and dropout
        scores = scores.softmax();
        attn_cache = scores;
        Tensor scores_2d_attn;

        if (dropout_rate > 0.0f && training) {
            // Reshape scores to [batch_size * num_heads, seq_len, seq_len] for dropout
            std::vector<float> scores_2d_data(batch_size * num_heads * seq_len * seq_len);
            for (int n = 0; n < batch_size; ++n) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int i = 0; i < seq_len; ++i) {
                        for (int j = 0; j < seq_len; ++j) {
                            int idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                            int src_idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            scores_2d_data[idx] = scores.data[src_idx];
                        }
                    }
                }
            }
            Tensor scores_2d_for_dropout(scores_2d_data, {batch_size * num_heads, seq_len, seq_len});
            mask = scores_2d_for_dropout.dropout_mask(dropout_rate, rng);

            //std::cout << "MHSA mask shape: {";
            //for (int s : mask.shape) std::cout << s << ",";
            //std::cout << "}\n";


            scores_2d_attn = scores_2d_for_dropout * mask;
        } else {
            scores_2d_attn = Tensor(scores_2d.data, {batch_size * num_heads, seq_len, seq_len});
        }

        // Compute attention output
        std::vector<float> V_attn(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        V_attn[idx] = V.data[src_idx];
                    }
                }
            }
        }
        Tensor V_2d(V_attn, {batch_size * num_heads, seq_len, head_dim});

        Tensor out_2d = scores_2d_attn.matmul(V_2d);

        // Reshape output to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> out_data(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        out_data[idx] = out_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor out(out_data, {batch_size, seq_len, num_heads, head_dim});

        // Reshape to [batch_size * seq_len, embed_dim]
        std::vector<float> final_out_data(batch_size * seq_len * embed_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        final_out_data[idx] = out.data[src_idx];
                    }
                }
            }
        }
        Tensor out_2d_final(final_out_data, {batch_size * seq_len, embed_dim});
        out_2d_final_cache = out_2d_final; // Cache for backward
        Tensor result = out_2d_final.matmul(W_o) + bias_o;

        // Reshape to [batch_size, seq_len, embed_dim]
        std::vector<float> output_data(batch_size * seq_len * embed_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int d = 0; d < embed_dim; ++d) {
                    int idx = n * seq_len * embed_dim + p * embed_dim + d;
                    int src_idx = (n * seq_len + p) * embed_dim + d;
                    output_data[idx] = result.data[src_idx];
                }
            }
        }
        Tensor output(output_data, {batch_size, seq_len, embed_dim});

        //std::cout << "MHSA output shape: {";
        //for (int s : output.shape) std::cout << s << ",";
        //std::cout << "}\n";
        
        return output;
    }
    Tensor backward(const Tensor& grad_output) override {
        int batch_size = grad_output.shape[0];
        int seq_len = grad_output.shape[1];
        int head_dim = embed_dim / num_heads;

        // Reshape grad_output to [batch_size * seq_len, embed_dim]
        Tensor grad_output_2d(grad_output.data, {batch_size * seq_len, embed_dim});

        // Backward through W_o and bias_o
        W_o_grad = out_2d_final_cache.transpose().matmul(grad_output_2d);
        std::vector<float> bias_o_grad_data(embed_dim, 0.0f);
        for (int i = 0; i < batch_size * seq_len; ++i) {
            for (int d = 0; d < embed_dim; ++d) {
                bias_o_grad_data[d] += grad_output_2d.data[i * embed_dim + d];
            }
        }
        bias_o_grad = Tensor(bias_o_grad_data, {embed_dim});
        Tensor grad_out_2d = grad_output_2d.matmul(W_o.transpose());

        // Reshape grad_out_2d to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> grad_out_data(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        grad_out_data[idx] = grad_out_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_out(grad_out_data, {batch_size, seq_len, num_heads, head_dim});

        // Reshape grad_out to [batch_size * num_heads, seq_len, head_dim]
        std::vector<float> grad_out_2d_data(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        grad_out_2d_data[idx] = grad_out.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_out_2d_reshaped(grad_out_2d_data, {batch_size * num_heads, seq_len, head_dim});

        // Reshape attn_cache to [batch_size * num_heads, seq_len, seq_len]
        std::vector<float> attn_2d_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                        int src_idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        attn_2d_data[idx] = attn_cache.data[src_idx];
                    }
                }
            }
        }
        Tensor attn_2d(attn_2d_data, {batch_size * num_heads, seq_len, seq_len});

        // Backward through attention: grad_scores_2d = grad_out_2d_reshaped * attn_2d^T
        Tensor grad_scores_2d = attn_2d.transpose().matmul(grad_out_2d_reshaped);

        //std::cout << "Backward grad_scores_2d shape: {";
        //for (int s : grad_scores_2d.shape) std::cout << s << ",";
        //std::cout << "} mask shape: {";
        //for (int s : mask.shape) std::cout << s << ",";
        //std::cout << "}\n";

        // Apply dropout mask
        if (dropout_rate > 0.0f) {
            grad_scores_2d = grad_scores_2d * mask;
        }

        // Softmax backward
        std::vector<float> grad_attn_2d_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size * num_heads; ++n) {
            for (int i = 0; i < seq_len; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    int idx = n * seq_len * seq_len + i * seq_len + j;
                    sum += attn_2d.data[idx] * grad_scores_2d.data[idx];
                }
                for (int j = 0; j < seq_len; ++j) {
                    int idx = n * seq_len * seq_len + i * seq_len + j;
                    grad_attn_2d_data[idx] = grad_scores_2d.data[idx] - sum * attn_2d.data[idx];
                }
            }
        }
        Tensor grad_attn_2d(grad_attn_2d_data, {batch_size * num_heads, seq_len, seq_len});

        // Reshape grad_attn_2d to [batch_size, num_heads, seq_len, seq_len]
        std::vector<float> grad_attn_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        int src_idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                        grad_attn_data[idx] = grad_attn_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_attn(grad_attn_data, {batch_size, num_heads, seq_len, seq_len});

        // Reshape q_cache, k_cache, v_cache to [batch_size * num_heads, seq_len, head_dim]
        std::vector<float> q_2d_data(batch_size * num_heads * seq_len * head_dim);
        std::vector<float> k_2d_data(batch_size * num_heads * seq_len * head_dim);
        std::vector<float> v_2d_data(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        q_2d_data[idx] = q_cache.data[src_idx];
                        k_2d_data[idx] = k_cache.data[src_idx];
                        v_2d_data[idx] = v_cache.data[src_idx];
                    }
                }
            }
        }
        Tensor Q_2d(q_2d_data, {batch_size * num_heads, seq_len, head_dim});
        Tensor K_2d(k_2d_data, {batch_size * num_heads, seq_len, head_dim});
        Tensor V_2d(v_2d_data, {batch_size * num_heads, seq_len, head_dim});

        // Backward through Q, K, V
        Tensor grad_Q_2d = grad_attn_2d.matmul(K_2d) / std::sqrt((float)head_dim);
        Tensor grad_K_2d = grad_attn_2d.transpose().matmul(Q_2d) / std::sqrt((float)head_dim);
        Tensor grad_V_2d = attn_2d.matmul(grad_out_2d_reshaped);

        // Reshape gradients to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> grad_Q_data(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> grad_K_data(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> grad_V_data(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        grad_Q_data[idx] = grad_Q_2d.data[src_idx];
                        grad_K_data[idx] = grad_K_2d.data[src_idx];
                        grad_V_data[idx] = grad_V_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_Q(grad_Q_data, {batch_size, seq_len, num_heads, head_dim});
        Tensor grad_K(grad_K_data, {batch_size, seq_len, num_heads, head_dim});
        Tensor grad_V(grad_V_data, {batch_size, seq_len, num_heads, head_dim});

        // Reshape to [batch_size * seq_len, embed_dim]
        std::vector<float> grad_Q_2d_final(batch_size * seq_len * embed_dim);
        std::vector<float> grad_K_2d_final(batch_size * seq_len * embed_dim);
        std::vector<float> grad_V_2d_final(batch_size * seq_len * embed_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        grad_Q_2d_final[idx] = grad_Q.data[src_idx];
                        grad_K_2d_final[idx] = grad_K.data[src_idx];
                        grad_V_2d_final[idx] = grad_V.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_Q_final(grad_Q_2d_final, {batch_size * seq_len, embed_dim});
        Tensor grad_K_final(grad_K_2d_final, {batch_size * seq_len, embed_dim});
        Tensor grad_V_final(grad_V_2d_final, {batch_size * seq_len, embed_dim});

        // Compute gradients for weights
        Tensor input_2d(input_cache.data, {batch_size * seq_len, embed_dim});
        W_q_grad = input_2d.transpose().matmul(grad_Q_final);
        W_k_grad = input_2d.transpose().matmul(grad_K_final);
        W_v_grad = input_2d.transpose().matmul(grad_V_final);
        std::vector<float> bias_q_grad_data(embed_dim, 0.0f);
        std::vector<float> bias_k_grad_data(embed_dim, 0.0f);
        std::vector<float> bias_v_grad_data(embed_dim, 0.0f);
        for (int i = 0; i < batch_size * seq_len; ++i) {
            for (int d = 0; d < embed_dim; ++d) {
                bias_q_grad_data[d] += grad_Q_final.data[i * embed_dim + d];
                bias_k_grad_data[d] += grad_K_final.data[i * embed_dim + d];
                bias_v_grad_data[d] += grad_V_final.data[i * embed_dim + d];
            }
        }
        bias_q_grad = Tensor(bias_q_grad_data, {embed_dim});
        bias_k_grad = Tensor(bias_k_grad_data, {embed_dim});
        bias_v_grad = Tensor(bias_v_grad_data, {embed_dim});

        // Compute input gradient
        Tensor grad_input_2d = grad_Q_final.matmul(W_q.transpose()) + grad_K_final.matmul(W_k.transpose()) + grad_V_final.matmul(W_v.transpose());

        // Reshape to [batch_size, seq_len, embed_dim]
        Tensor grad_input(grad_input_2d.data, {batch_size, seq_len, embed_dim});
        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(W_q, W_q_grad);
        optimizer->update(W_k, W_k_grad);
        optimizer->update(W_v, W_v_grad);
        optimizer->update(W_o, W_o_grad);
        optimizer->update(bias_q, bias_q_grad);
        optimizer->update(bias_k, bias_k_grad);
        optimizer->update(bias_v, bias_v_grad);
        optimizer->update(bias_o, bias_o_grad);
    }

    size_t num_params() const override {
        return W_q.total_elements() + W_k.total_elements() + W_v.total_elements() + W_o.total_elements() +
               bias_q.total_elements() + bias_k.total_elements() + bias_v.total_elements() + bias_o.total_elements();
    }
};
class FeedForward : public Layer {
private:
    Tensor W1;
    Tensor b1;
    Tensor W2;
    Tensor b2;
    std::shared_ptr<Activation> activation;
    float dropout_rate;
    Tensor input_cache;
    Tensor z1_cache;
    Tensor act_cache;
    Tensor dropout_cache;
    std::mt19937 rng;
    Tensor W1_grad;
    Tensor b1_grad;
    Tensor W2_grad;
    Tensor b2_grad;

public:
    FeedForward(int embed_dim, int ff_dim, ActivationType act_type, float dropout_rate = 0.0f)
        : dropout_rate(dropout_rate), rng(std::random_device{}()) {
        switch (act_type) {
            case ActivationType::ReLU: activation = std::make_shared<ReLU>(); break;
            case ActivationType::Sigmoid: activation = std::make_shared<Sigmoid>(); break;
            default: throw std::runtime_error("Unsupported activation for FeedForward");
        }
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim));
        std::vector<float> w1_data(embed_dim * ff_dim);
        std::vector<float> w2_data(ff_dim * embed_dim);
        for (float& w : w1_data) w = dist(rng);
        for (float& w : w2_data) w = dist(rng);
        W1 = Tensor(w1_data, {embed_dim, ff_dim});
        W2 = Tensor(w2_data, {ff_dim, embed_dim});
        b1 = Tensor(std::vector<float>(ff_dim, 0.0f), {ff_dim});
        b2 = Tensor(std::vector<float>(embed_dim, 0.0f), {embed_dim});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 3 || input.shape[2] != W1.shape[0]) {
            throw std::runtime_error("Invalid input shape for FeedForward");
        }
        input_cache = input;
        Tensor z1 = input.matmul(W1) + b1;
        z1_cache = z1;
        Tensor act = activation->apply_batch(z1);
        act_cache = act;

        Tensor out = act.matmul(W2) + b2;
        if (training && dropout_rate > 0.0f) {
            std::vector<float> mask_data(out.data.size());
            std::bernoulli_distribution dist(1.0f - dropout_rate);
            for (size_t i = 0; i < out.data.size(); ++i) {
                mask_data[i] = dist(rng) ? 1.0f / (1.0f - dropout_rate) : 0.0f;
            }
            dropout_cache = Tensor(mask_data, out.shape);
            out = out * dropout_cache;
        } else if (!training && dropout_rate > 0.0f) {
            std::vector<float> out_data(out.data.size());
            for (size_t i = 0; i < out.data.size(); ++i) {
                out_data[i] = out.data[i] * (1.0f - dropout_rate);
            }
            out = Tensor(out_data, out.shape);
        }
        return out;
    }

    Tensor backward(const Tensor& grad_output) override {
    // Gradiente después de dropout (si lo hubo)
    Tensor grad_out = dropout_rate > 0.0f ? grad_output * dropout_cache : grad_output;

    // Reshape: aplanamos batch * seq_len
    int batch = grad_out.shape[0];
    int seq_len = grad_out.shape[1];
    int embed_dim = grad_out.shape[2];
    int ff_dim = W1.shape[1];

    Tensor act_flat = act_cache.reshape({batch * seq_len, ff_dim});     // (B*S, FF)
    Tensor grad_flat = grad_out.reshape({batch * seq_len, embed_dim}); // (B*S, E)

    // Gradiente W2 y b2
    W2_grad = act_flat.transpose().matmul(grad_flat); // (FF, B*S) x (B*S, E) -> (FF, E)

    std::vector<float> b2_grad_data(embed_dim, 0.0f);
    for (int i = 0; i < grad_flat.shape[0]; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            b2_grad_data[j] += grad_flat.data[i * embed_dim + j];
        }
    }
    b2_grad = Tensor(b2_grad_data, {embed_dim});

    // Gradiente después de W2
    Tensor grad_act = grad_flat.matmul(W2.transpose()); // (B*S, FF)

    // Derivada de la activación
    Tensor act_deriv = activation->derivative_batch(z1_cache, act_cache); // (B, S, FF)
    Tensor act_deriv_flat = act_deriv.reshape({batch * seq_len, ff_dim});

    Tensor grad_z1 = act_deriv_flat * grad_act; // (B*S, FF)

    // Gradiente W1 y b1
    Tensor input_flat = input_cache.reshape({batch * seq_len, embed_dim});
    W1_grad = input_flat.transpose().matmul(grad_z1); // (E, FF)

    std::vector<float> b1_grad_data(ff_dim, 0.0f);
    for (int i = 0; i < grad_z1.shape[0]; ++i) {
        for (int j = 0; j < ff_dim; ++j) {
            b1_grad_data[j] += grad_z1.data[i * ff_dim + j];
        }
    }
    b1_grad = Tensor(b1_grad_data, {ff_dim});

    // Gradiente hacia atrás (input)
    Tensor grad_input_flat = grad_z1.matmul(W1.transpose()); // (B*S, E)
    return grad_input_flat.reshape({batch, seq_len, embed_dim});
}


    void update_weights(Optimizer* optimizer) override {
        optimizer->update(W1, W1_grad);
        optimizer->update(b1, b1_grad);
        optimizer->update(W2, W2_grad);
        optimizer->update(b2, b2_grad);
    }

    size_t num_params() const override {
        return W1.total_elements() + b1.size() + W2.total_elements() + b2.size();
    }
};

class TransformerEncoder : public Layer {
private:
    std::shared_ptr<LayerNorm> ln1;
    std::shared_ptr<MultiHeadSelfAttention> mhsa;
    std::shared_ptr<LayerNorm> ln2;
    std::shared_ptr<FeedForward> ff;
    Tensor mhsa_cache;
    Tensor ff_cache;

public:
    TransformerEncoder(int embed_dim, int num_heads, int ff_dim, float dropout_rate, ActivationType act_type)
        : ln1(std::make_shared<LayerNorm>(embed_dim)),
          mhsa(std::make_shared<MultiHeadSelfAttention>(embed_dim, num_heads, dropout_rate)),
          ln2(std::make_shared<LayerNorm>(embed_dim)),
          ff(std::make_shared<FeedForward>(embed_dim, ff_dim, act_type, dropout_rate)) {}

    Tensor forward(const Tensor& input, bool training = false) override {
        Tensor x = ln1->forward(input, training);
        mhsa_cache = mhsa->forward(x, training);

        //std::cout << "TransformerEncoder: input shape {";
        //for (int s : input.shape) std::cout << s << ",";
        //std::cout << "}, mhsa_cache shape {";
        //for (int s : mhsa_cache.shape) std::cout << s << ",";
        //std::cout << "}\n";
        
        x = input + mhsa_cache;
        ff_cache = ff->forward(ln2->forward(x, training), training);
        
        //std::cout << "TransformerEncoder: x shape {";
        //for (int s : x.shape) std::cout << s << ",";
        //std::cout << "}, ff_cache shape {";
        //for (int s : ff_cache.shape) std::cout << s << ",";
        //std::cout << "}\n";

        return x + ff_cache;
    }
/*
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_ff = ln2->backward(grad_output);
        grad_ff = ff->backward(grad_ff);
        Tensor grad_x = grad_output + grad_ff;
        grad_x = ln1->backward(mhsa->backward(grad_x));
        return grad_x + grad_output;
    }*/
    Tensor backward(const Tensor& grad_output) override {
        // Paso 1: Gradiente con respecto a la suma final (x + FF)
        Tensor grad_ff_out = grad_output;

        // Paso 2: Retropropagación a través del bloque FFN
        Tensor grad_ff_in = ff->backward(grad_ff_out);           // dL/d(ff_input)
        Tensor grad_ln2 = ln2->backward(grad_ff_in);             // dL/d(post_mhsa)

        // Paso 3: Sumar con el skip connection de la salida de MHSA
        Tensor grad_post_mhsa = grad_ln2 + grad_output;

        // Paso 4: Retropropagación a través del MHSA
        Tensor grad_mhsa = mhsa->backward(grad_post_mhsa);       // dL/d(LN1_out)
        Tensor grad_ln1 = ln1->backward(grad_mhsa);              // dL/d(input)

        // Nota: no es necesario sumar otra vez grad_output aquí, ya se usó
        return grad_ln1;
    }

    void update_weights(Optimizer* optimizer) override {
        ln1->update_weights(optimizer);
        mhsa->update_weights(optimizer);
        ln2->update_weights(optimizer);
        ff->update_weights(optimizer);
    }

    size_t num_params() const override {
        return ln1->num_params() + mhsa->num_params() + ln2->num_params() + ff->num_params();
    }
};


class FilterTokenizer : public Layer {
private:
    std::shared_ptr<Dense> linear1;  // (C -> L)
    std::shared_ptr<Dense> linear2;  // (C -> D)

    Tensor attn_weights;     // softmax result: (N, L, HW)
    Tensor input_cache;      // original input: (N, HW, C)
    Tensor a_cache;          // output of linear1 before softmax: (N, HW, L)
    Tensor softmax_cache;    // after softmax: (N, HW, L)
    Tensor token_cache;      // final output: (N, L, D)

public:
    FilterTokenizer(int in_channels, int token_channels, int num_tokens) {
        // Linear1: C → L  (token scores)
        linear1 = std::make_shared<Dense>(in_channels, num_tokens, ActivationType::None);
        // Linear2: C → D  (project token embeddings)
        linear2 = std::make_shared<Dense>(in_channels, token_channels, ActivationType::None);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        // input shape: (N, HW, C)
        input_cache = input;

        Tensor a = linear1->forward(input, training);  // (N, HW, L)
        a_cache = a;

        softmax_cache = a.softmax();  // softmax along HW dimension (dim=1)

        Tensor attn = softmax_cache.transpose(1, 2);  // (N, L, HW)

        attn_weights = attn;

        Tensor weighted_sum = attn.matmul(input);  // (N, L, C)

        token_cache = linear2->forward(weighted_sum, training);  // (N, L, D)

        return token_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        // grad_output: (N, L, D)
        Tensor grad_linear2 = linear2->backward(grad_output);  // (N, L, C)

        // Now compute grad w.r.t attention weights
        // attn: (N, L, HW), input: (N, HW, C)
        Tensor grad_attn = grad_linear2.matmul(input_cache.transpose(1, 2));  // (N, L, HW)

        Tensor grad_softmax = grad_attn.transpose(1, 2);  // (N, HW, L)

        // backprop through softmax
        // Use standard formula: dL/da = softmax * (dL/dy - sum_j(dL/dy_j * softmax_j))
        Tensor sum = (grad_softmax * softmax_cache).sum(2).reshape({softmax_cache.shape[0], softmax_cache.shape[1], 1});
        Tensor d_softmax = softmax_cache * (grad_softmax - sum);  // (N, HW, L)

        Tensor grad_linear1 = linear1->backward(d_softmax);  // (N, HW, C)

        Tensor grad_input = attn_weights.transpose(1, 2).matmul(grad_linear2);  // (N, HW, C)

        // total gradient wrt input
        return grad_linear1 + grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        linear1->update_weights(optimizer);
        linear2->update_weights(optimizer);
    }

    size_t num_params() const override {
        return linear1->num_params() + linear2->num_params();
    }
};

class BatchNorm1D : public Layer {
private:
    Tensor running_mean;
    Tensor running_var;
    Tensor gamma;
    Tensor beta;
    Tensor input_cache;
    Tensor norm_cache;

    float momentum;
    float eps;
    bool is_initialized;

    Tensor dgamma;
    Tensor dbeta;

public:
    BatchNorm1D(int num_features, float momentum_ = 0.1f, float eps_ = 1e-5f)
        : momentum(momentum_), eps(eps_), is_initialized(false) {
        gamma = Tensor(std::vector<float>(num_features, 1.0f), {num_features});
        beta  = Tensor(std::vector<float>(num_features, 0.0f), {num_features});
        running_mean = Tensor(std::vector<float>(num_features, 0.0f), {num_features});
        running_var  = Tensor(std::vector<float>(num_features, 1.0f), {num_features});
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        input_cache = input;
        int batch_size = input.shape[0];
        int features = input.shape[1];

        std::vector<float> mean(features, 0.0f);
        std::vector<float> var(features, 0.0f);

        // Calcular media y varianza por canal
        for (int f = 0; f < features; ++f) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                sum += input.data[b * features + f];
            }
            mean[f] = sum / batch_size;

            float sq_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                float diff = input.data[b * features + f] - mean[f];
                sq_sum += diff * diff;
            }
            var[f] = sq_sum / batch_size;
        }

        // Actualizar medias en entrenamiento
        if (training) {
            for (int i = 0; i < features; ++i) {
                running_mean.data[i] = momentum * mean[i] + (1.0f - momentum) * running_mean.data[i];
                running_var.data[i]  = momentum * var[i]  + (1.0f - momentum) * running_var.data[i];
            }
        } else {
            mean = running_mean.data;
            var = running_var.data;
        }

        // Normalizar
        std::vector<float> normed_data(batch_size * features);
        for (int b = 0; b < batch_size; ++b) {
            for (int f = 0; f < features; ++f) {
                float x = input.data[b * features + f];
                float normalized = (x - mean[f]) / std::sqrt(var[f] + eps);
                normed_data[b * features + f] = gamma.data[f] * normalized + beta.data[f];
            }
        }

        norm_cache = Tensor(normed_data, input.shape);
        return norm_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        // Simplificada: no implementa derivada exacta de batchnorm
        // Para entrenamiento real, se debería usar autodiff o implementación completa
        return grad_output;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(gamma, dgamma);
        optimizer->update(beta, dbeta);
    }

    size_t num_params() const override {
        return gamma.total_elements() + beta.total_elements();
    }
};


class Projector : public Layer {
private:
    Dense proj_feature;  // linear1
    Dense proj_token_key;  // linear2
    Dense proj_token_value;  // linear3
    std::shared_ptr<Dense> downsample = nullptr;

    std::shared_ptr<BatchNorm1D> norm;
    std::shared_ptr<ReLU> relu = std::make_shared<ReLU>();

    Tensor cache_attn;
    int in_channels, out_channels, token_channels;

public:
    Projector(int in_channels_, int out_channels_, int token_channels_)
        : in_channels(in_channels_), out_channels(out_channels_), token_channels(token_channels_),
          proj_feature(in_channels_, token_channels_, ActivationType::None),
          proj_token_key(token_channels_, token_channels_, ActivationType::None),
          proj_token_value(token_channels_, out_channels_, ActivationType::None),
          norm(std::make_shared<BatchNorm1D>(out_channels_)) {

        if (in_channels != out_channels) {
            downsample = std::make_shared<Dense>(in_channels, out_channels, ActivationType::None);
        }
    }

    Tensor forward(const Tensor& x, const Tensor& tokens, bool training = false) {
        // x: (N, HW, C_in), tokens: (N, L, D)

        Tensor x_q = proj_feature.forward(x, training);        // (N, HW, token_channels)
        Tensor t_k = proj_token_key.forward(tokens, training); // (N, L, token_channels)
        Tensor t_v = proj_token_value.forward(tokens, training); // (N, L, out_channels)

        Tensor t_k_T = t_k.transpose(1, 2); // (N, token_channels, L)
        Tensor attn = x_q.matmul(t_k_T);   // (N, HW, L)
        attn = attn.softmax();             // softmax on last dim (L)
        cache_attn = attn;

        Tensor message = attn.matmul(t_v); // (N, HW, out_channels)

        Tensor x_proj = x;
        if (downsample) {
            x_proj = downsample->forward(x, training); // (N, HW, out_channels)
        }

        Tensor out = x_proj + message; // fusion
        out = out.transpose(1, 2);     // (N, C, HW) → para BN1D

        out = norm->forward(out, training);
        out = relu->apply_batch(out);

        out = out.transpose(1, 2);     // Volver a (N, HW, C)

        return out;
    }

    Tensor backward(const Tensor& grad_output) override {
        // grad_output: (N, HW, C_out) después de ReLU + BN
        Tensor grad = grad_output.transpose(1, 2); // (N, C_out, HW)

        // Paso 1: backward por ReLU y BatchNorm
        grad = relu->derivative_batch(norm->forward(grad_output.transpose(1, 2))) * grad;
        grad = norm->backward(grad);
        grad = grad.transpose(1, 2); // Volver a (N, HW, C_out)

        // Paso 2: split grad respecto a suma x_proj + message
        Tensor grad_message = grad;
        Tensor grad_x_proj = grad;

        // Paso 3: backward de downsample si existe
        Tensor grad_x;
        if (downsample) {
            grad_x = downsample->backward(grad_x_proj);
        } else {
            grad_x = grad_x_proj;
        }

        // Paso 4: backward sobre atención
        // attn: (N, HW, L), t_v: (N, L, C_out)
        Tensor grad_attn = grad_message.matmul(proj_token_value.forward(tokens).transpose(1, 2)); // (N, HW, L)
        Tensor grad_t_v = cache_attn.transpose(1, 2).matmul(grad_message); // (N, L, C_out)

        // Paso 5: backprop softmax
        Tensor softmax_input = proj_feature.forward(x).matmul(proj_token_key.forward(tokens).transpose(1, 2)); // x_q * t_k^T
        Tensor attn_softmax = cache_attn;
        Tensor sum = (grad_attn * attn_softmax).sum(2).reshape({attn_softmax.shape[0], attn_softmax.shape[1], 1});
        Tensor d_softmax = attn_softmax * (grad_attn - sum); // (N, HW, L)

        // Paso 6: backward matmul de atención: x_q * t_k^T
        Tensor x_q = proj_feature.forward(x);          // (N, HW, token_channels)
        Tensor t_k = proj_token_key.forward(tokens);   // (N, L, token_channels)

        Tensor grad_x_q = d_softmax.matmul(t_k);       // (N, HW, token_channels)
        Tensor grad_t_k = d_softmax.transpose(1, 2).matmul(x_q); // (N, L, token_channels)

        // Paso 7: backward proyecciones lineales
        Tensor grad_tokens_1 = proj_token_key.backward(grad_t_k);     // contribución de t_k
        Tensor grad_tokens_2 = proj_token_value.backward(grad_t_v);   // contribución de t_v
        Tensor grad_tokens_total = grad_tokens_1 + grad_tokens_2;

        Tensor grad_x_feature = proj_feature.backward(grad_x_q);      // grad respecto a x

        return grad_x + grad_x_feature;  // total grad_input
    }


    void update_weights(Optimizer* opt) override {
        proj_feature.update_weights(opt);
        proj_token_key.update_weights(opt);
        proj_token_value.update_weights(opt);
        if (downsample) downsample->update_weights(opt);
        norm->update_weights(opt);
    }

    size_t num_params() const override {
        return proj_feature.num_params() + proj_token_key.num_params() +
               proj_token_value.num_params() + (downsample ? downsample->num_params() : 0) +
               norm->num_params();
    }
};

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

class Dense : public Layer {
private:
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    Tensor z_cache;
    std::shared_ptr<Activation> activation;
    float weight_decay;
    Tensor weights_grad;
    Tensor bias_grad;
    std::mt19937 rng;

    void initialize_weights(int in_size, int out_size) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_size));
        std::vector<float> w_data(in_size * out_size);
        for (float& w : w_data) {
            w = dist(rng);
        }
        weights = Tensor(w_data, {in_size, out_size});
        std::vector<float> b_data(out_size, 0.0f);
        bias = Tensor(b_data, {out_size});
    }

public:
    Dense(int in_size, int out_size, ActivationType act_type, float wd = 0.0f)
        : weight_decay(wd), rng(std::random_device{}()) {
        switch (act_type) {
            case ActivationType::ReLU: activation = std::make_shared<ReLU>(); break;
            case ActivationType::Sigmoid: activation = std::make_shared<Sigmoid>(); break;
            case ActivationType::Softmax: activation = std::make_shared<Softmax>(); break;
            case ActivationType::None: activation = nullptr; break; // ✅ soporta sin activación
            default: throw std::runtime_error("Unsupported activation for Dense");
        }
        initialize_weights(in_size, out_size);
    }
    /*
        Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape[1] != weights.shape[0]) {
            throw std::runtime_error("Input shape mismatch");
        }
        input_cache = input;
        z_cache = input.matmul(weights) + bias;
        return activation->apply_batch(z_cache);
    }
    */
    
    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape[1] != weights.shape[0]) {
            throw std::runtime_error("Input shape mismatch");
        }
        input_cache = input;
        z_cache = input.matmul(weights) + bias;
        return (activation ? activation->apply_batch(z_cache) : z_cache); // ✅ sin activación si nullptr
    }
    /*
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_z = activation->derivative_batch(z_cache, activation->apply_batch(z_cache));
        for (size_t i = 0; i < grad_z.data.size(); ++i) {
            grad_z.data[i] *= grad_output.data[i];
        }

        weights_grad = input_cache.transpose().matmul(grad_z);
        if (weight_decay > 0.0f) {
            for (size_t i = 0; i < weights_grad.data.size(); ++i) {
                weights_grad.data[i] += 2.0f * weight_decay * weights.data[i];
            }
        }

        std::vector<float> bias_grad_data(bias.shape[0], 0.0f);
        for (int i = 0; i < grad_z.shape[0]; ++i) {
            for (int j = 0; j < bias.shape[0]; ++j) {
                bias_grad_data[j] += grad_z.data[i * bias.shape[0] + j];
            }
        }
        bias_grad = Tensor(bias_grad_data, bias.shape);

        return grad_z.matmul(weights.transpose());
    }/-
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_z;

        // ⚠️ Si la activación es Softmax, asumimos CrossEntropyLoss y usamos directamente grad_output
        if (dynamic_cast<Softmax*>(activation.get()) != nullptr) {
            grad_z = grad_output; // ya es y_pred - y_true
        } else {
            Tensor activated = activation->apply_batch(z_cache);
            grad_z = activation->derivative_batch(z_cache, activated);
            for (size_t i = 0; i < grad_z.data.size(); ++i) {
                grad_z.data[i] *= grad_output.data[i];
            }
        }

        weights_grad = input_cache.transpose().matmul(grad_z);

        if (weight_decay > 0.0f) {
            for (size_t i = 0; i < weights_grad.data.size(); ++i) {
                weights_grad.data[i] += 2.0f * weight_decay * weights.data[i];
            }
        }

        std::vector<float> bias_grad_data(bias.shape[0], 0.0f);
        for (int i = 0; i < grad_z.shape[0]; ++i) {
            for (int j = 0; j < bias.shape[0]; ++j) {
                bias_grad_data[j] += grad_z.data[i * bias.shape[0] + j];
            }
        }
        bias_grad = Tensor(bias_grad_data, bias.shape);

        return grad_z.matmul(weights.transpose());
    }*/

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_z;

        if (!activation) {
            grad_z = grad_output;  // ✅ sin activación, pasa grad_output directo
        } else if (dynamic_cast<Softmax*>(activation.get()) != nullptr) {
            grad_z = grad_output; // ⚠️ asumiendo CrossEntropy
        } else {
            Tensor activated = activation->apply_batch(z_cache);
            grad_z = activation->derivative_batch(z_cache, activated);
            for (size_t i = 0; i < grad_z.data.size(); ++i) {
                grad_z.data[i] *= grad_output.data[i];
            }
        }

        weights_grad = input_cache.transpose().matmul(grad_z);

        if (weight_decay > 0.0f) {
            for (size_t i = 0; i < weights_grad.data.size(); ++i) {
                weights_grad.data[i] += 2.0f * weight_decay * weights.data[i];
            }
        }

        std::vector<float> bias_grad_data(bias.shape[0], 0.0f);
        for (int i = 0; i < grad_z.shape[0]; ++i) {
            for (int j = 0; j < bias.shape[0]; ++j) {
                bias_grad_data[j] += grad_z.data[i * bias.shape[0] + j];
            }
        }
        bias_grad = Tensor(bias_grad_data, bias.shape);

        return grad_z.matmul(weights.transpose());
    }


    void update_weights(Optimizer* optimizer) override {
        optimizer->update(weights, weights_grad);
        optimizer->update(bias, bias_grad);
    }

    size_t num_params() const override {
        return weights.total_elements() + bias.total_elements();
    }
};

class Loss {
public:
    virtual float compute(const Tensor& y_pred, const Tensor& y_true) const = 0;
    virtual Tensor gradient(const Tensor& y_pred, const Tensor& y_true) const = 0;
    virtual ~Loss() {}
};

class CrossEntropyLoss : public Loss {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) const override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss computation");
        }
        float loss = 0.0f;
        int batch_size = y_pred.shape[0];
        int num_classes = y_pred.shape[1];

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                loss -= y_true.data[idx] * std::log(y_pred.data[idx] + 1e-10f);
            }
        }
        return loss / batch_size;
    }

    Tensor gradient(const Tensor& y_pred, const Tensor& y_true) const override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss gradient");
        }
        std::vector<float> grad_data(y_pred.data.size());
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            grad_data[i] = y_pred.data[i] - y_true.data[i];
        }
        return Tensor(grad_data, y_pred.shape);
    }
};

class Metric {
public:
    virtual float compute(const Tensor& y_pred, const Tensor& y_true) const = 0;
    virtual std::string name() const = 0;
    virtual ~Metric() {}
};

class Accuracy : public Metric {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) const override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in accuracy computation");
        }
        int batch_size = y_pred.shape[0];
        int num_classes = y_pred.shape[1];
        int correct = 0;

        for (int i = 0; i < batch_size; ++i) {
            int pred_class = 0;
            int true_class = 0;
            float max_pred = y_pred.data[i * num_classes];
            float max_true = y_true.data[i * num_classes];

            for (int j = 1; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                if (y_pred.data[idx] > max_pred) {
                    max_pred = y_pred.data[idx];
                    pred_class = j;
                }
                if (y_true.data[idx] > max_true) {
                    max_true = y_true.data[idx];
                    true_class = j;
                }
            }
            if (pred_class == true_class) {
                ++correct;
            }
        }
        return static_cast<float>(correct) / batch_size;
    }

    std::string name() const override { return "accuracy"; }
};

class Logger {
public:
    virtual void log(const std::string& message) {
        std::cout << message << std::endl;
    }
    virtual ~Logger() {}
};


// Model class
class Model {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<Loss> loss;
    std::shared_ptr<Optimizer> optimizer;
    std::vector<std::shared_ptr<Metric>> metrics;
    std::shared_ptr<Logger> logger;
    bool training_mode;

public:
    Model() : training_mode(false) {}

    size_t num_params() const {
        return std::accumulate(layers.begin(), layers.end(), size_t(0),
            [](size_t total, const std::shared_ptr<Layer>& layer) {
                return total + layer->num_params();
            });
    }

    void add(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    void add_metric(std::shared_ptr<Metric> metric) {
        metrics.push_back(metric);
    }

    void compile(std::shared_ptr<Loss> loss_fn, std::shared_ptr<Optimizer> opt,
                 std::shared_ptr<Logger> log = nullptr) {
        loss = loss_fn;
        optimizer = opt;
        logger = log;
    }

    Tensor forward(const Tensor& input, bool training = false) {
        Tensor current = input;
        for (auto& layer : layers) {
            current = layer->forward(current, training);
        }
        return current;
    }

    void evaluate(const Tensor& X, const Tensor& y, int batch_size = 32) {
        if (X.shape[0] != y.shape[0]) {
            throw std::runtime_error("Input and target shape mismatch in evaluation");
        }

        float eval_loss = 0.0f;
        std::vector<float> eval_metrics(metrics.size(), 0.0f);
        int num_batches = (X.shape[0] + batch_size - 1) / batch_size;

        training_mode = false;

        for (int b = 0; b < num_batches; ++b) {
            int start = b * batch_size;
            int end = std::min(start + batch_size, X.shape[0]);
            int current_batch_size = end - start;
            
            std::vector<float> batch_data;
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < X.total_elements() / X.shape[0]; ++j) {
                    batch_data.push_back(X.data[i * (X.total_elements() / X.shape[0]) + j]);
                }
            }
            std::vector<float> batch_target;
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < y.shape[1]; ++j) {
                    batch_target.push_back(y.data[i * y.shape[1] + j]);
                }
            }

            Tensor X_batch(batch_data, {current_batch_size, X.shape[1], X.shape[2], X.shape[3]});
            Tensor y_batch(batch_target, {current_batch_size, y.shape[1]});

            Tensor y_pred = forward(X_batch, false);

            eval_loss += loss->compute(y_pred, y_batch);

            for (size_t m = 0; m < metrics.size(); ++m) {
                eval_metrics[m] += metrics[m]->compute(y_pred, y_batch);
            }
        }

        eval_loss /= num_batches;
        for (float& m : eval_metrics) {
            m /= num_batches;
        }

        if (logger) {
            std::string log_message = "Evaluation - Loss: " + std::to_string(eval_loss);
            for (size_t m = 0; m < metrics.size(); ++m) {
                log_message += ", " + metrics[m]->name() + ": " + std::to_string(eval_metrics[m]);
            }
            logger->log(log_message);
        }
    }

    void fit(const Tensor& X, const Tensor& y, const Tensor& X_test, const Tensor& y_test, int epochs, int batch_size = 32) {
        if (X.shape[0] != y.shape[0]) {
            throw std::runtime_error("Input and target shape mismatch in training");
        }
        if (X_test.shape[0] != y_test.shape[0]) {
            throw std::runtime_error("Input and target shape mismatch in test set");
        }

        std::vector<int> indices(X.shape[0]);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(std::random_device{}());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            
            float epoch_loss = 0.0f;
            std::vector<float> epoch_metrics(metrics.size(), 0.0f);
            int num_batches = (X.shape[0] + batch_size - 1) / batch_size;

            training_mode = true;

            for (int b = 0; b < num_batches; ++b) {
                int start = b * batch_size;
                int end = std::min(start + batch_size, X.shape[0]);
                int current_batch_size = end - start;
                
                std::vector<float> batch_data;
                for (int i = start; i < end; ++i) {
                    int idx = indices[i];
                    for (int j = 0; j < X.total_elements() / X.shape[0]; ++j) {
                        batch_data.push_back(X.data[idx * (X.total_elements() / X.shape[0]) + j]);
                    }
                }
                std::vector<float> batch_target;
                for (int i = start; i < end; ++i) {
                    int idx = indices[i];
                    for (int j = 0; j < y.shape[1]; ++j) {
                        batch_target.push_back(y.data[idx * y.shape[1] + j]);
                    }
                }

                Tensor X_batch(batch_data, {current_batch_size, X.shape[1], X.shape[2], X.shape[3]});
                Tensor y_batch(batch_target, {current_batch_size, y.shape[1]});

                Tensor y_pred = forward(X_batch, true);

                float batch_loss = loss->compute(y_pred, y_batch);
                epoch_loss += batch_loss;

                for (size_t m = 0; m < metrics.size(); ++m) {
                    epoch_metrics[m] += metrics[m]->compute(y_pred, y_batch);
                }

                Tensor grad = loss->gradient(y_pred, y_batch);
                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    grad = (*it)->backward(grad);
                }

                for (auto& layer : layers) {
                    layer->update_weights(optimizer.get());
                }
            }

            epoch_loss /= num_batches;
            for (float& m : epoch_metrics) {
                m /= num_batches;
            }

            if (logger) {
                std::string log_message = "Epoch " + std::to_string(epoch + 1) + "/" + 
                                         std::to_string(epochs) + ", Train Loss: " + 
                                         std::to_string(epoch_loss);
                for (size_t m = 0; m < metrics.size(); ++m) {
                    log_message += ", Train " + metrics[m]->name() + ": " + 
                                  std::to_string(epoch_metrics[m]);
                }
                logger->log(log_message);
            }

            evaluate(X_test, y_test, batch_size);
        }
    }
};

// Model Builder class
class ModelBuilder {
private:
    std::vector<std::shared_ptr<Layer>> layers;

    std::shared_ptr<Activation> get_activation(ActivationType type) {
        switch (type) {
            case ActivationType::ReLU: return std::make_shared<ReLU>();
            case ActivationType::Sigmoid: return std::make_shared<Sigmoid>();
            case ActivationType::Softmax: return std::make_shared<Softmax>();
            default: throw std::runtime_error("Unknown activation type");
        }
    }

public:
    ModelBuilder& add_patch_embedding(int in_channels, int patch_size, int embed_dim) {
        layers.push_back(std::make_shared<PatchEmbedding>(in_channels, patch_size, embed_dim));
        return *this;
    }

    ModelBuilder& add_positional_encoding(int num_patches, int embed_dim, bool learnable = false) {
        layers.push_back(std::make_shared<PositionalEncoding>(num_patches, embed_dim, learnable));
        return *this;
    }

    ModelBuilder& add_transformer_encoder(int embed_dim, int num_heads, int ff_dim,
                                         float dropout_rate, ActivationType act_type = ActivationType::ReLU) {
        layers.push_back(std::make_shared<TransformerEncoder>(embed_dim, num_heads, ff_dim, dropout_rate, act_type));
        return *this;
    }

    ModelBuilder& add_dense(int in_size, int out_size, ActivationType activation = ActivationType::ReLU, float weight_decay = 0.0f) {
        layers.push_back(std::make_shared<Dense>(in_size, out_size, activation, weight_decay));
        return *this;
    }

    ModelBuilder& add_dropout(float rate) {
        layers.push_back(std::make_shared<Dropout>(rate));
        return *this;
    }

    ModelBuilder& add_global_average_pooling() {
        layers.push_back(std::make_shared<GlobalAveragePooling>());
        return *this;
    }

    ModelBuilder& add_flatten() {
        layers.push_back(std::make_shared<Flatten>());
        return *this;
    }


    std::shared_ptr<Model> build() {
        auto model = std::make_shared<Model>();
        for (auto& layer : layers) {
            model->add(layer);
        }
        return model;
    }
};

