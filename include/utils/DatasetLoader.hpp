#pragma once

#include "core/Tensor.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <filesystem>

namespace DataLoader {

inline int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

inline Tensor load_images(const std::string& path, int num_images = -1) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) throw std::runtime_error("Invalid image file!");

    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    number_of_images = reverse_int(number_of_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    if (num_images > 0 && num_images < number_of_images) number_of_images = num_images;

    Tensor tensor({number_of_images, 1, rows, cols});
    for (int i = 0; i < number_of_images; ++i) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                tensor.at({i, 0, r, c}) = static_cast<float>(temp) / 255.0f;
            }
        }
    }
    return tensor;
}

inline Tensor load_labels(const std::string& path, int num_labels = -1) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

    int magic_number = 0;
    int number_of_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2049) throw std::runtime_error("Invalid label file!");

    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);
    if (num_labels > 0 && num_labels < number_of_labels) number_of_labels = num_labels;

    Tensor labels({number_of_labels, 10});
    labels.fill(0.0f);

    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels.at({i, static_cast<int>(temp)}) = 1.0f; // one-hot encoding
    }
    return labels;
}

inline void load_mnist_data(Tensor& train_X, Tensor& train_y, Tensor& test_X, Tensor& test_y, int num_train = 1000, int num_test = 1000) {
    std::string base = "data/";

    train_X = load_images(base + "train-images.idx3-ubyte", num_train);
    train_y = load_labels(base + "train-labels.idx1-ubyte", num_train);

    test_X = load_images(base + "t10k-images.idx3-ubyte", num_test);
    test_y = load_labels(base + "t10k-labels.idx1-ubyte", num_test);
}

} // namespace DataLoader
