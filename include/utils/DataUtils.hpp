#pragma once

#include "core/Tensor.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>

namespace utils {

inline void replicate_channels(Tensor& batch, int channels) {
    if (batch.shape.size() != 4 || batch.shape[1] != 1)
        throw std::runtime_error("Expected shape [N, 1, H, W]");

    int N = batch.shape[0];
    int H = batch.shape[2];
    int W = batch.shape[3];
    int HW = H * W;

    std::vector<float> new_data(N * channels * HW);

    for (int n = 0; n < N; ++n) {
        const float* src = batch.data.data() + n * HW;
        for (int c = 0; c < channels; ++c) {
            float* dst = new_data.data() + (n * channels + c) * HW;
            std::copy(src, src + HW, dst);
        }
    }

    batch.data = std::move(new_data);
    batch.shape = {N, channels, H, W};
}

inline void train_val_split(const Tensor& X, const Tensor& y, float val_ratio,
                            Tensor& X_train, Tensor& y_train,
                            Tensor& X_val, Tensor& y_val) {
    if (X.shape[0] != y.shape[0])
        throw std::runtime_error("Shape mismatch between X and y");

    int total = X.shape[0];
    int val_size = static_cast<int>(val_ratio * total);
    int train_size = total - val_size;

    std::vector<int> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

    auto extract = [](const Tensor& full, const std::vector<int>& idxs, int from, int to) {
        int count = to - from;
        int per_sample = full.total_elements() / full.shape[0];
        std::vector<float> out_data;
        for (int i = from; i < to; ++i) {
            int idx = idxs[i];
            out_data.insert(out_data.end(),
                            full.data.begin() + idx * per_sample,
                            full.data.begin() + (idx + 1) * per_sample);
        }
        auto shape = full.shape;
        shape[0] = count;
        return Tensor(out_data, shape);
    };

    X_train = extract(X, indices, 0, train_size);
    y_train = extract(y, indices, 0, train_size);
    X_val   = extract(X, indices, train_size, total);
    y_val   = extract(y, indices, train_size, total);
}

} // namespace utils
