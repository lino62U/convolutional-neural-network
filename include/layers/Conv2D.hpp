#pragma once

#include "core/Layer.hpp"
#include "core/Tensor.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>
#include <algorithm>
#include <string>


enum class PaddingType { VALID, SAME, CUSTOM };

class Conv2D : public Layer {
private:
    Tensor filters, bias;
    Tensor filters_grad, bias_grad;

    PaddingType padding_type;
    int custom_pad = 0;
    int stride = 1;
    int computed_padding = 0;

    std::shared_ptr<Activation> activation;

    Tensor input_cache;
    Tensor z_cache;
    Tensor a_cache;

    std::mt19937 rng;

    void initialize_filters(int in_channels, int num_filters, int kernel_size) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / (in_channels * kernel_size * kernel_size)));
        std::vector<float> f_data(num_filters * in_channels * kernel_size * kernel_size);
        for (auto& f : f_data) f = dist(rng);
        filters = Tensor(std::move(f_data), {num_filters, in_channels, kernel_size, kernel_size});
        bias = Tensor(std::vector<float>(num_filters, 0.0f), {num_filters});
    }

public:
    Conv2D(int in_channels, int num_filters, int kernel_size,
           PaddingType pad_type = PaddingType::VALID, int stride_ = 1,
           std::shared_ptr<Activation> act = nullptr)
        : padding_type(pad_type), stride(stride_), activation(std::move(act)), rng(std::random_device{}()) {
        initialize_filters(in_channels, num_filters, kernel_size);
    }

    void set_custom_padding(int pad) {
        padding_type = PaddingType::CUSTOM;
        custom_pad = pad;
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        const int N = input.shape[0];
        const int C = input.shape[1];
        const int H = input.shape[2];
        const int W = input.shape[3];
        const int K = filters.shape[2];
        const int Kw = filters.shape[3];
        const int F = filters.shape[0];

        // Padding
        int pad = 0;
        if (padding_type == PaddingType::SAME)
            pad = ((H - 1) * stride + K - H) / 2;
        else if (padding_type == PaddingType::CUSTOM)
            pad = custom_pad;
        computed_padding = pad;

        input_cache = input.pad(pad);

        const int Hp = input_cache.shape[2];
        const int Wp = input_cache.shape[3];
        const int Oh = (Hp - K) / stride + 1;
        const int Ow = (Wp - Kw) / stride + 1;

        z_cache = Tensor({N, F, Oh, Ow});
        float* z_data = z_cache.data.data();
        const float* in_data = input_cache.data.data();
        const float* f_data = filters.data.data();
        const float* b_data = bias.data.data();

        for (int n = 0; n < N; ++n)
            for (int f = 0; f < F; ++f)
                for (int h = 0; h < Oh; ++h)
                    for (int w = 0; w < Ow; ++w) {
                        float sum = b_data[f];
                        for (int c = 0; c < C; ++c)
                            for (int kh = 0; kh < K; ++kh)
                                for (int kw = 0; kw < Kw; ++kw) {
                                    int ih = h * stride + kh;
                                    int iw = w * stride + kw;
                                    int in_idx = ((n * C + c) * Hp + ih) * Wp + iw;
                                    int filt_idx = ((f * C + c) * K + kh) * Kw + kw;
                                    sum += in_data[in_idx] * f_data[filt_idx];
                                }
                        int out_idx = ((n * F + f) * Oh + h) * Ow + w;
                        z_data[out_idx] = sum;
                    }

        if (activation) {
            a_cache = activation->forward(z_cache, training);
            return a_cache;
        }
        return z_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        const Tensor& grad_z = activation ? activation->backward(grad_output) : grad_output;

        const int N = input_cache.shape[0];
        const int C = input_cache.shape[1];
        const int Hp = input_cache.shape[2];
        const int Wp = input_cache.shape[3];
        const int F = filters.shape[0];
        const int K = filters.shape[2];
        const int Kw = filters.shape[3];
        const int Oh = grad_z.shape[2];
        const int Ow = grad_z.shape[3];

        filters_grad = Tensor::zeros(filters.shape);
        bias_grad = Tensor::zeros(bias.shape);
        Tensor grad_input(input_cache.shape);
        grad_input.fill(0.0f);

        float* grad_in_data = grad_input.data.data();
        const float* gradz_data = grad_z.data.data();
        const float* in_data = input_cache.data.data();
        float* filt_grad_data = filters_grad.data.data();
        float* bias_grad_data = bias_grad.data.data();
        const float* filt_data = filters.data.data();

        for (int n = 0; n < N; ++n)
            for (int f = 0; f < F; ++f)
                for (int h = 0; h < Oh; ++h)
                    for (int w = 0; w < Ow; ++w) {
                        float d_out = gradz_data[((n * F + f) * Oh + h) * Ow + w];
                        bias_grad_data[f] += d_out;

                        for (int c = 0; c < C; ++c)
                            for (int kh = 0; kh < K; ++kh)
                                for (int kw = 0; kw < Kw; ++kw) {
                                    int ih = h * stride + kh;
                                    int iw = w * stride + kw;
                                    int in_idx = ((n * C + c) * Hp + ih) * Wp + iw;
                                    int filt_idx = ((f * C + c) * K + kh) * Kw + kw;
                                    filt_grad_data[filt_idx] += in_data[in_idx] * d_out;
                                    grad_in_data[in_idx] += filt_data[filt_idx] * d_out;
                                }
                    }

        return grad_input.unpad(computed_padding);
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(filters, filters_grad);
        optimizer->update(bias, bias_grad);
    }

    size_t num_params() const override {
        return filters.total_elements() + bias.total_elements();
    }
};