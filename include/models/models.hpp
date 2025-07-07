#pragma once

#include "NeuralNet.hpp"
#include <memory>
#include <iostream>

Model build_cnn_model() {
    Model model;

    // 28×28×3 → 28×28×16
    auto conv1 = std::make_shared<Conv2D>(3, 16, 3, PaddingType::CUSTOM, 1);
    conv1->set_custom_padding(1);
    model.add(conv1);
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<MaxPooling2D>(2, 2));  // 28→14

    // 14×14×16 → 14×14×4
    auto conv2 = std::make_shared<Conv2D>(16, 4, 3, PaddingType::CUSTOM, 1);
    conv2->set_custom_padding(1);
    model.add(conv2);
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<MaxPooling2D>(2, 2));  // 14→7

    model.add(std::make_shared<Flatten>());           // 7×7×64 → 3136
    model.add(std::make_shared<Linear>(196, 16));
    model.add(std::make_shared<ReLU>());
     model.add(std::make_shared<Linear>(196, 10));
    model.add(std::make_shared<Softmax>());

    return model;
}

Model build_mlp_model() {
    Model model;

    // Entrada: 28×28×3 = 2352
    model.add(std::make_shared<Flatten>());

    // Capa 1: 2352 → 42 = 98,784 + 42 = 98,826
    model.add(std::make_shared<Linear>(28 * 28 * 3, 64));
    model.add(std::make_shared<Dropout>(0.25f));  // Dropout con p = 0.25
    model.add(std::make_shared<ReLU>());

    // Capa 2: 42 → 24 = 42×24 + 24 = 1032
    model.add(std::make_shared<Linear>(64, 32));
    model.add(std::make_shared<Dropout>(0.25f));
    model.add(std::make_shared<ReLU>());

    // Capa 3: 24 → 10 = 240 + 10 = 250
    model.add(std::make_shared<Linear>(32, 10));
    model.add(std::make_shared<Softmax>());

    return model;
}

