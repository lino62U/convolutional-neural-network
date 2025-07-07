#pragma once

#include "NeuralNet.hpp"
#include <memory>
#include <iostream>

// Construye un modelo de red neuronal convolucional (CNN)
Model build_cnn_model() {
    Model model;

    // Primera capa convolucional: Entrada 28×28×3 → Salida 28×28×16
    auto conv1 = std::make_shared<Conv2D>(3, 16, 3, PaddingType::CUSTOM, 1);
    conv1->set_custom_padding(1);  // Padding personalizado para mantener dimensiones
    model.add(conv1);
    model.add(std::make_shared<ReLU>());  // Activación ReLU
    model.add(std::make_shared<MaxPooling2D>(2, 2));  // Reducción de dimensiones: 28×28 → 14×14

    // Segunda capa convolucional: Entrada 14×14×16 → Salida 14×14×4
    auto conv2 = std::make_shared<Conv2D>(16, 4, 3, PaddingType::CUSTOM, 1);
    conv2->set_custom_padding(1);  // Padding personalizado para mantener dimensiones
    model.add(conv2);
    model.add(std::make_shared<ReLU>());  // Activación ReLU
    model.add(std::make_shared<MaxPooling2D>(2, 2));  // Reducción de dimensiones: 14×14 → 7×7

    // Aplanamiento: Entrada 7×7×4 → Salida 3136 (vector plano)
    model.add(std::make_shared<Flatten>());

    // Capa totalmente conectada: Entrada 3136 → Salida 16
    model.add(std::make_shared<Linear>(196, 16));
    model.add(std::make_shared<ReLU>());  // Activación ReLU

    // Capa de salida: Entrada 16 → Salida 10 (clases)
    model.add(std::make_shared<Linear>(196, 10));
    model.add(std::make_shared<Softmax>());  // Activación Softmax para clasificación

    return model;
}

// Construye un modelo de red neuronal multicapa (MLP)
Model build_mlp_model() {
    Model model;

    // Aplanamiento de entrada: Entrada 28×28×3 → Salida 2352 (vector plano)
    model.add(std::make_shared<Flatten>());

    // Primera capa totalmente conectada: Entrada 2352 → Salida 64
    model.add(std::make_shared<Linear>(28 * 28 * 3, 64));
    model.add(std::make_shared<Dropout>(0.25f));  // Dropout con probabilidad de 0.25
    model.add(std::make_shared<ReLU>());  // Activación ReLU

    // Segunda capa totalmente conectada: Entrada 64 → Salida 32
    model.add(std::make_shared<Linear>(64, 32));
    model.add(std::make_shared<Dropout>(0.25f));  // Dropout con probabilidad de 0.25
    model.add(std::make_shared<ReLU>());  // Activación ReLU

    // Capa de salida: Entrada 32 → Salida 10 (clases)
    model.add(std::make_shared<Linear>(32, 10));
    model.add(std::make_shared<Softmax>());  // Activación Softmax para clasificación

    return model;
}
