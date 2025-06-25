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
enum class ActivationType { ReLU, Sigmoid, Softmax };

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
        if (input.shape.size() != 2) {
            throw std::runtime_error("Softmax expects 2D input tensor");
        }
        int batch_size = input.shape[0];
        int num_classes = input.shape[1];
        std::vector<float> output(input.data.size());

        for (int i = 0; i < batch_size; ++i) {
            float max_val = *std::max_element(
                input.data.begin() + i * num_classes,
                input.data.begin() + (i + 1) * num_classes
            );
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                output[idx] = std::exp(input.data[idx] - max_val);
                sum_exp += output[idx];
            }
            for (int j = 0; j < num_classes; ++j) {
                output[i * num_classes + j] /= sum_exp;
            }
        }
        return Tensor(output, input.shape);
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

class SGD : public Optimizer {
private:
    float learning_rate;
    float momentum;
    std::vector<Tensor> velocity;

public:
    SGD(float lr = 0.01f, float mom = 0.9f) : learning_rate(lr), momentum(mom) {}

    void update(Tensor& param, const Tensor& grad) override {
        if (param.shape != grad.shape) {
            throw std::runtime_error("Shape mismatch in optimizer update");
        }

        if (velocity.empty() || velocity[0].shape != param.shape) {
            velocity.clear();
            velocity.emplace_back(std::vector<float>(param.size(), 0.0f), param.shape);
        }

        for (size_t i = 0; i < param.data.size(); ++i) {
            velocity[0].data[i] = momentum * velocity[0].data[i] - learning_rate * grad.data[i];
            param.data[i] += velocity[0].data[i];
        }
    }
};

class Adam : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    std::vector<Tensor> m;
    std::vector<Tensor> v;

public:
    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void update(Tensor& param, const Tensor& grad) override {
        if (param.shape != grad.shape) {
            throw std::runtime_error("Shape mismatch in optimizer update");
        }

        if (m.empty() || m[0].shape != param.shape) {
            m.clear();
            v.clear();
            m.emplace_back(std::vector<float>(param.size(), 0.0f), param.shape);
            v.emplace_back(std::vector<float>(param.size(), 0.0f), param.shape);
        }

        ++t;

        for (size_t i = 0; i < param.data.size(); ++i) {
            m[0].data[i] = beta1 * m[0].data[i] + (1.0f - beta1) * grad.data[i];
            v[0].data[i] = beta2 * v[0].data[i] + (1.0f - beta2) * grad.data[i] * grad.data[i];
        }

        std::vector<float> m_hat(param.size()), v_hat(param.size());
        float beta1_t = std::pow(beta1, t);
        float beta2_t = std::pow(beta2, t);
        for (size_t i = 0; i < param.data.size(); ++i) {
            m_hat[i] = m[0].data[i] / (1.0f - beta1_t);
            v_hat[i] = v[0].data[i] / (1.0f - beta2_t);
            param.data[i] -= learning_rate * m_hat[i] / (std::sqrt(v_hat[i]) + epsilon);
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

class Convolution : public Layer {
private:
    Tensor filters;
    Tensor bias;
    int padding;
    int stride;
    float weight_decay;
    std::shared_ptr<Activation> activation;
    Tensor input_cache;
    Tensor z_cache;
    Tensor activation_cache;
    Tensor filters_grad;
    Tensor bias_grad;
    std::mt19937 rng;

    void initialize_filters(int in_channels, int num_filters, int kernel_size) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / (in_channels * kernel_size * kernel_size)));
        std::vector<float> f_data(num_filters * in_channels * kernel_size * kernel_size);
        for (float& f : f_data) {
            f = dist(rng);
        }
        filters = Tensor(f_data, {num_filters, in_channels, kernel_size, kernel_size});
        
        std::vector<float> b_data(num_filters, 0.0f);
        bias = Tensor(b_data, {num_filters});
    }

public:
    Convolution(int in_channels, int num_filters, int kernel_size, int pad, int str, 
                std::shared_ptr<Activation> act, float wd = 0.0f)
        : padding(pad), stride(str), weight_decay(wd), activation(act), rng(std::random_device{}()) {
        if (pad < 0 || str <= 0 || kernel_size <= 0 || wd < 0.0f) {
            throw std::runtime_error("Invalid convolution parameters");
        }
        initialize_filters(in_channels, num_filters, kernel_size);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 4 || input.shape[1] != filters.shape[1]) {
            throw std::runtime_error("Invalid input shape for convolution");
        }

        int batch_size = input.shape[0];
        int in_channels = input.shape[1];
        int in_height = input.shape[2];
        int in_width = input.shape[3];
        int num_filters = filters.shape[0];
        int kernel_height = filters.shape[2];
        int kernel_width = filters.shape[3];

        int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        if (out_height <= 0 || out_width <= 0) {
            throw std::runtime_error("Invalid output dimensions");
        }

        std::vector<float> padded_input_data(batch_size * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding), 0.0f);
        Tensor padded_input(padded_input_data, {batch_size, in_channels, in_height + 2 * padding, in_width + 2 * padding});
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        padded_input.data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                         c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                         (h + padding) * (in_width + 2 * padding) + (w + padding)] =
                            input.data[n * in_channels * in_height * in_width + c * in_height * in_width + h * in_width + w];
                    }
                }
            }
        }

        input_cache = padded_input;

        std::vector<float> output_data(batch_size * num_filters * out_height * out_width);
        for (int n = 0; n < batch_size; ++n) {
            for (int f = 0; f < num_filters; ++f) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        float sum = bias.data[f];
                        for (int c = 0; c < in_channels; ++c) {
                            for (int kh = 0; kh < kernel_height; ++kh) {
                                for (int kw = 0; kw < kernel_width; ++kw) {
                                    int input_h = h * stride + kh;
                                    int input_w = w * stride + kw;
                                    sum += padded_input.data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                            c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                            input_h * (in_width + 2 * padding) + input_w] *
                                           filters.data[f * in_channels * kernel_height * kernel_width +
                                                        c * kernel_height * kernel_width + kh * kernel_width + kw];
                                }
                            }
                        }
                        output_data[n * num_filters * out_height * out_width + f * out_height * out_width + h * out_width + w] = sum;
                    }
                }
            }
        }

        z_cache = Tensor(output_data, {batch_size, num_filters, out_height, out_width});
        activation_cache = activation->apply_batch(z_cache);
        return activation_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape != activation_cache.shape) {
            throw std::runtime_error("Gradient shape mismatch in convolution backward");
        }

        int batch_size = input_cache.shape[0];
        int in_channels = input_cache.shape[1];
        int in_height = input_cache.shape[2] - 2 * padding;
        int in_width = input_cache.shape[3] - 2 * padding;
        int num_filters = filters.shape[0];
        int kernel_height = filters.shape[2];
        int kernel_width = filters.shape[3];
        int out_height = grad_output.shape[2];
        int out_width = grad_output.shape[3];

        Tensor grad_z = activation->derivative_batch(z_cache, activation_cache);
        for (size_t i = 0; i < grad_z.data.size(); ++i) {
            grad_z.data[i] *= grad_output.data[i];
        }

        std::vector<float> grad_filters_data(filters.size(), 0.0f);
        for (int f = 0; f < num_filters; ++f) {
            for (int c = 0; c < in_channels; ++c) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        float sum = 0.0f;
                        for (int n = 0; n < batch_size; ++n) {
                            for (int h = 0; h < out_height; ++h) {
                                for (int w = 0; w < out_width; ++w) {
                                    sum += input_cache.data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                           c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                           (h * stride + kh) * (in_width + 2 * padding) + (w * stride + kw)] *
                                           grad_z.data[n * num_filters * out_height * out_width +
                                                       f * out_height * out_width + h * out_width + w];
                                }
                            }
                        }
                        int idx = f * in_channels * kernel_height * kernel_width + c * kernel_height * kernel_width + kh * kernel_width + kw;
                        grad_filters_data[idx] = sum;
                        if (weight_decay > 0.0f) {
                            grad_filters_data[idx] += 2.0f * weight_decay * filters.data[idx];
                        }
                    }
                }
            }
        }
        filters_grad = Tensor(grad_filters_data, filters.shape);

        std::vector<float> grad_bias_data(num_filters, 0.0f);
        for (int f = 0; f < num_filters; ++f) {
            for (int n = 0; n < batch_size; ++n) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        grad_bias_data[f] += grad_z.data[n * num_filters * out_height * out_width +
                                                        f * out_height * out_width + h * out_width + w];
                    }
                }
            }
        }
        bias_grad = Tensor(grad_bias_data, bias.shape);

        std::vector<float> grad_input_data(input_cache.size(), 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        for (int f = 0; f < num_filters; ++f) {
                            for (int kh = 0; kh < kernel_height; ++kh) {
                                for (int kw = 0; kw < kernel_width; ++kw) {
                                    int out_h = (h - kh + padding) / stride;
                                    int out_w = (w - kw + padding) / stride;
                                    if (out_h >= 0 && out_h < out_height && (h - kh + padding) % stride == 0 &&
                                        out_w >= 0 && out_w < out_width && (w - kw + padding) % stride == 0) {
                                        grad_input_data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                       c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                                       (h + padding) * (in_width + 2 * padding) + (w + padding)] +=
                                            grad_z.data[n * num_filters * out_height * out_width +
                                                       f * out_height * out_width + out_h * out_width + out_w] *
                                            filters.data[f * in_channels * kernel_height * kernel_width +
                                                        c * kernel_height * kernel_width + kh * kernel_width + kw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::vector<float> grad_input_unpadded(batch_size * in_channels * in_height * in_width);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        grad_input_unpadded[n * in_channels * in_height * in_width + c * in_height * in_width + h * in_width + w] =
                            grad_input_data[n * in_channels * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                           c * (in_height + 2 * padding) * (in_width + 2 * padding) +
                                           (h + padding) * (in_width + 2 * padding) + (w + padding)];
                    }
                }
            }
        }

        return Tensor(grad_input_unpadded, {batch_size, in_channels, in_height, in_width});
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(filters, filters_grad);
        optimizer->update(bias, bias_grad);
    }

    size_t num_params() const override {
        return filters.total_elements() + bias.total_elements();
    }
};

class MaxPooling : public Layer {
private:
    int pool_size;
    int stride;
    Tensor input_cache;
    std::vector<std::vector<int>> max_indices;

public:
    MaxPooling(int size, int str) : pool_size(size), stride(str) {
        if (size <= 0 || str <= 0) {
            throw std::runtime_error("Invalid max pooling parameters");
        }
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape.size() != 4) {
            throw std::runtime_error("Invalid input shape for max pooling");
        }

        int batch_size = input.shape[0];
        int channels = input.shape[1];
        int in_height = input.shape[2];
        int in_width = input.shape[3];

        int out_height = (in_height - pool_size) / stride + 1;
        int out_width = (in_width - pool_size) / stride + 1;
        if (out_height <= 0 || out_width <= 0) {
            throw std::runtime_error("Invalid output dimensions for max pooling");
        }

        input_cache = input;
        max_indices.clear();
        max_indices.resize(batch_size * channels * out_height * out_width);

        std::vector<float> output_data(batch_size * channels * out_height * out_width);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_idx = 0;
                        for (int ph = 0; ph < pool_size; ++ph) {
                            for (int pw = 0; pw < pool_size; ++pw) {
                                int input_h = h * stride + ph;
                                int input_w = w * stride + pw;
                                if (input_h < in_height && input_w < in_width) {
                                    float val = input.data[n * channels * in_height * in_width +
                                                          c * in_height * in_width + input_h * in_width + input_w];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = ph * pool_size + pw;
                                    }
                                }
                            }
                        }
                        int output_idx = n * channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
                        output_data[output_idx] = max_val;
                        max_indices[output_idx] = std::vector<int>{n, c, h * stride + max_idx / pool_size, w * stride + max_idx % pool_size};
                    }
                }
            }
        }

        return Tensor(output_data, {batch_size, channels, out_height, out_width});
    }

    Tensor backward(const Tensor& grad_output) override {
        if (grad_output.shape[2] != (input_cache.shape[2] - pool_size) / stride + 1 ||
            grad_output.shape[3] != (input_cache.shape[3] - pool_size) / stride + 1) {
            throw std::runtime_error("Gradient shape mismatch in max pooling backward");
        }

        int batch_size = input_cache.shape[0];
        int channels = input_cache.shape[1];
        int in_height = input_cache.shape[2];
        int in_width = input_cache.shape[3];
        int out_height = grad_output.shape[2];
        int out_width = grad_output.shape[3];

        std::vector<float> grad_input_data(batch_size * channels * in_height * in_width, 0.0f);
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        int output_idx = n * channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
                        const auto& max_pos = max_indices[output_idx];
                        int input_h = max_pos[2];
                        int input_w = max_pos[3];
                        grad_input_data[max_pos[0] * channels * in_height * in_width +
                                       max_pos[1] * in_height * in_width + input_h * in_width + input_w] +=
                            grad_output.data[output_idx];
                    }
                }
            }
        }

        return Tensor(grad_input_data, input_cache.shape);
    }

    void update_weights(Optimizer* optimizer) override {}
    size_t num_params() const override { return 0; }
};

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

    void update_weights(Optimizer* optimizer) override {}
    size_t num_params() const override { return 0; }

private:
    Tensor input_cache;
};

class Dense : public Layer {
private:
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    Tensor z_cache;
    Tensor activation_cache;
    std::shared_ptr<Activation> activation;
    float weight_decay;
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
    Dense(int in_size, int out_size, std::shared_ptr<Activation> act, float wd = 0.0f)
        : activation(act), weight_decay(wd), rng(std::random_device{}()) {
        if (wd < 0.0f) {
            throw std::runtime_error("Weight decay must be non-negative");
        }
        initialize_weights(in_size, out_size);
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        if (input.shape[1] != weights.shape[0]) {
            throw std::runtime_error("Input shape mismatch");
        }
        input_cache = input;
        z_cache = input.matmul(weights) + bias;
        activation_cache = activation->apply_batch(z_cache);
        return activation_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_z = activation->derivative_batch(z_cache, activation_cache);
        for (size_t i = 0; i < grad_z.data.size(); ++i) {
            grad_z.data[i] *= grad_output.data[i];
        }

        Tensor grad_weights = input_cache.transpose().matmul(grad_z);
        
        if (weight_decay > 0.0f) {
            for (size_t i = 0; i < grad_weights.data.size(); ++i) {
                grad_weights.data[i] += 2.0f * weight_decay * weights.data[i];
            }
        }

        std::vector<float> grad_bias_data(bias.shape[0], 0.0f);
        int batch_size = grad_z.shape[0];
        int out_size = grad_z.shape[1];
        for (int j = 0; j < out_size; ++j) {
            for (int i = 0; i < batch_size; ++i) {
                grad_bias_data[j] += grad_z.data[i * out_size + j];
            }
        }
        Tensor grad_bias(grad_bias_data, bias.shape);
        
        Tensor grad_input = grad_z.matmul(weights.transpose());

        weights_grad = grad_weights;
        bias_grad = grad_bias;

        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
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

class Loss {
public:
    virtual float compute(const Tensor& y_pred, const Tensor& y_true) const = 0;
    virtual Tensor gradient(const Tensor& y_pred, const Tensor& y_true) const = 0;
    virtual ~Loss() {}
};

class MSELoss : public Loss {
public:
    float compute(const Tensor& y_pred, const Tensor& y_true) const override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss computation");
        }
        float sum = 0.0f;
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            float diff = y_pred.data[i] - y_true.data[i];
            sum += diff * diff;
        }
        return sum / y_pred.data.size();
    }

    Tensor gradient(const Tensor& y_pred, const Tensor& y_true) const override {
        if (y_pred.shape != y_true.shape) {
            throw std::runtime_error("Shape mismatch in loss gradient");
        }
        std::vector<float> grad_data(y_pred.data.size());
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            grad_data[i] = 2.0f * (y_pred.data[i] - y_true.data[i]) / y_pred.data.size();
        }
        return Tensor(grad_data, y_pred.shape);
    }
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
    ModelBuilder& add_convolution(int in_channels, int num_filters, int kernel_size, int padding, int stride,
                                  ActivationType activation = ActivationType::ReLU, float weight_decay = 0.0f) {
        layers.push_back(std::make_shared<Convolution>(in_channels, num_filters, kernel_size, padding, stride,
                                                      get_activation(activation), weight_decay));
        return *this;
    }

    ModelBuilder& add_max_pooling(int pool_size, int stride) {
        layers.push_back(std::make_shared<MaxPooling>(pool_size, stride));
        return *this;
    }

    ModelBuilder& add_flatten() {
        layers.push_back(std::make_shared<Flatten>());
        return *this;
    }

    ModelBuilder& add_dense(int in_size, int out_size, ActivationType activation = ActivationType::ReLU, float weight_decay = 0.0f) {
        layers.push_back(std::make_shared<Dense>(in_size, out_size, get_activation(activation), weight_decay));
        return *this;
    }

    ModelBuilder& add_dropout(float rate) {
        layers.push_back(std::make_shared<Dropout>(rate));
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