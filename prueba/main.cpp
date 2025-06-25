#include <iostream>
//#include "neuronal.hpp"
#include "heders.hpp"


int main() {
    // Load MNIST training dataset with 1000 samples
    MNISTLoader train_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1000);

    // Load MNIST test dataset with 1000 samples
    MNISTLoader test_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000);

    // Create CNN model
    Model model;
    model.add(std::make_shared<Convolution>(1, 16, 3, 1, 1, std::make_shared<ReLU>())); // 28x28x1 -> 28x28x16
    model.add(std::make_shared<MaxPooling>(2, 2)); // 28x28x16 -> 14x14x16

    model.add(std::make_shared<Dropout>(0.3f)); // Dropout 30%


    model.add(std::make_shared<Convolution>(16, 32, 3, 1, 1, std::make_shared<ReLU>())); // 14x14x16 -> 14x14x32
    model.add(std::make_shared<MaxPooling>(2, 2)); // 14x14x32 -> 7x7x32
    model.add(std::make_shared<Flatten>()); // 7x7x32 -> 1568
    model.add(std::make_shared<Dense>(1568, 128, std::make_shared<ReLU>())); // 1568 -> 128

     model.add(std::make_shared<Dropout>(0.5f)); // Dropout 50%
    model.add(std::make_shared<Dense>(128, 10, std::make_shared<Softmax>())); // 128 -> 10

    // Add accuracy metric
    model.add_metric(std::make_shared<Accuracy>());

    // Compile with Cross-Entropy loss and Adam optimizer
    model.compile(std::make_shared<CrossEntropyLoss>(), 
                  std::make_shared<Adam>(0.001f), 
                  std::make_shared<Logger>());

    // Train with evaluation
    model.fit(train_data.images, train_data.labels, 
              test_data.images, test_data.labels, 
              10, 32);

    return 0;
}

/*
int main() {
    // Load MNIST training dataset with 1000 samples
    MNISTLoader train_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1000);

    // Load MNIST test dataset with 1000 samples
    MNISTLoader test_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000);

    // Create CNN model
    Model model;
    model.add(std::make_shared<Convolution>(1, 16, 3, 1, 1, std::make_shared<ReLU>(), 0.001f)); // 28x28x1 -> 28x28x16
    model.add(std::make_shared<MaxPooling>(2, 2)); // 28x28x16 -> 14x14x16
    model.add(std::make_shared<Dropout>(0.3f)); // Dropout 30%
    model.add(std::make_shared<Convolution>(16, 32, 3, 1, 1, std::make_shared<ReLU>(), 0.001f)); // 14x14x16 -> 14x14x32
    model.add(std::make_shared<MaxPooling>(2, 2)); // 14x14x32 -> 7x7x32
    model.add(std::make_shared<Flatten>()); // 7x7x32 -> 1568
    model.add(std::make_shared<Dense>(1568, 128, std::make_shared<ReLU>(), 0.001f)); // 1568 -> 128
    model.add(std::make_shared<Dropout>(0.5f)); // Dropout 50%
    model.add(std::make_shared<Dense>(128, 10, std::make_shared<Softmax>(), 0.0f)); // 128 -> 10, no weight decay

    // Add accuracy metric
    model.add_metric(std::make_shared<Accuracy>());

    // Compile with Cross-Entropy loss and Adam optimizer
    model.compile(std::make_shared<CrossEntropyLoss>(), 
                  std::make_shared<Adam>(0.001f), 
                  std::make_shared<Logger>());

    // Train with evaluation
    model.fit(train_data.images, train_data.labels, 
              test_data.images, test_data.labels, 
              10, 32);

    return 0;
}*/