#pragma once

#include "core/Tensor.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <filesystem>


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
    // Read big-endian 32-bit integer
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

        // Read header
        int magic = read_int32(file);
        int num_images = read_int32(file);
        int rows = read_int32(file);
        int cols = read_int32(file);

        if (magic != 2051) {
            throw std::runtime_error("Invalid MNIST image file format");
        }

        // Limit number of images if max_samples is specified
        if (max_samples > 0 && max_samples < num_images) {
            num_images = max_samples;
        }

        // Read image data
        std::vector<float> data(num_images * rows * cols);
        for (int i = 0; i < num_images * rows * cols; ++i) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            data[i] = pixel / 255.0f; // Normalize to [0, 1]
        }
        file.close();

        // Reshape to {batch, channels=1, height, width}
        images = Tensor(data, {num_images, 1, rows, cols});
    }

    void load_labels(const std::string& filename, int max_samples) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open label file: " + filename);
        }

        // Read header
        int magic = read_int32(file);
        int num_labels = read_int32(file);

        if (magic != 2049) {
            throw std::runtime_error("Invalid MNIST label file format");
        }

        // Limit number of labels if max_samples is specified
        if (max_samples > 0 && max_samples < num_labels) {
            num_labels = max_samples;
        }

        // Read labels and convert to one-hot encoding
        std::vector<float> data(num_labels * 10, 0.0f); // 10 classes
        for (int i = 0; i < num_labels; ++i) {
            unsigned char label;
            file.read(reinterpret_cast<char*>(&label), 1);
            data[i * 10 + label] = 1.0f;
        }
        file.close();

        labels = Tensor(data, {num_labels, 10});
    }
};