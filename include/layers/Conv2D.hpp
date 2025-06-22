#pragma once

#include "core/Layer.hpp"
#include "core/Tensor.hpp"
#include "optimizers/Optimizer.hpp"
#include <stdexcept>

class Conv2D : public Layer {
private:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    std::string padding_type;

    Tensor filters;
    Tensor bias;
    Tensor input_cache;
    Tensor grad_filters;
    Tensor grad_bias;

public:
    Conv2D(int in_ch, int out_ch, int k_h, int k_w, int s_h = 1, int s_w = 1, const std::string& padding = "valid")
        : in_channels(in_ch), out_channels(out_ch),
          kernel_h(k_h), kernel_w(k_w),
          stride_h(s_h), stride_w(s_w),
          padding_type(padding) {

        filters = Tensor({out_ch, in_ch, k_h, k_w});
        bias = Tensor({out_ch});
        grad_filters = Tensor({out_ch, in_ch, k_h, k_w});
        grad_bias = Tensor({out_ch});
    }

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        int batch = input.shape[0];
        int in_c = input.shape[1];
        int in_h = input.shape[2];
        int in_w = input.shape[3];

        int pad_h = 0, pad_w = 0;
        if (padding_type == "same") {
            pad_h = std::max((out_size(in_h, kernel_h, stride_h, "same") - 1) * stride_h + kernel_h - in_h, 0) / 2;
            pad_w = std::max((out_size(in_w, kernel_w, stride_w, "same") - 1) * stride_w + kernel_w - in_w, 0) / 2;
        }

        int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

        Tensor output({batch, out_channels, out_h, out_w});
        output.fill(0.0f);

        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float sum = bias.at({oc});
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int ih = oh * stride_h + kh - pad_h;
                                    int iw = ow * stride_w + kw - pad_w;

                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        sum += input.at({b, ic, ih, iw}) *
                                               filters.at({oc, ic, kh, kw});
                                    }
                                }
                            }
                        }
                        output.at({b, oc, oh, ow}) = sum;
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        const Tensor& input = input_cache;
        int batch = input.shape[0];
        int in_h = input.shape[2], in_w = input.shape[3];
        int out_h = grad_output.shape[2], out_w = grad_output.shape[3];

        int pad_h = 0, pad_w = 0;
        if (padding_type == "same") {
            pad_h = std::max((out_size(in_h, kernel_h, stride_h, "same") - 1) * stride_h + kernel_h - in_h, 0) / 2;
            pad_w = std::max((out_size(in_w, kernel_w, stride_w, "same") - 1) * stride_w + kernel_w - in_w, 0) / 2;
        }

        Tensor grad_input(input.shape);
        grad_input.fill(0.0f);
        grad_filters.fill(0.0f);
        grad_bias.fill(0.0f);

        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float grad_val = grad_output.at({b, oc, oh, ow});
                        grad_bias.at({oc}) += grad_val;

                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int ih = oh * stride_h + kh - pad_h;
                                    int iw = ow * stride_w + kw - pad_w;

                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        grad_input.at({b, ic, ih, iw}) += filters.at({oc, ic, kh, kw}) * grad_val;
                                        grad_filters.at({oc, ic, kh, kw}) += input.at({b, ic, ih, iw}) * grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(filters, grad_filters);
        optimizer->update(bias, grad_bias);
    }

    Tensor& get_filters() { return filters; }
    Tensor& get_bias() { return bias; }
    Tensor& get_grad_bias() { return grad_bias; }


private:
    int out_size(int in_size, int kernel, int stride, const std::string& pad) const {
        if (pad == "same") return (in_size + stride - 1) / stride;
        return (in_size - kernel) / stride + 1;
    }
};
