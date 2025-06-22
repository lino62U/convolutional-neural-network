#include "NeuralNet.hpp"
#include <iostream>
#include <memory>

int main() {
    // Crear modelo CNN para MNIST
    Model model;

    // Capa convolucional: 1 canal de entrada, 8 filtros de 3x3
    model.add(std::make_shared<Conv2D>(1, 8, 3, 3));
    model.add(std::make_shared<ReLU>());

    // Capa densa 1: salida 128 neuronas (después de flatten)
    model.add(std::make_shared<Flatten>()); // (faltaría implementar esta clase)
    model.add(std::make_shared<Dense>(8 * 26 * 26, 128, new XavierInitializer()));
    model.add(std::make_shared<ReLU>());

    // Capa de salida con 10 clases (dígitos del 0 al 9)
    model.add(std::make_shared<Dense>(128, 10, new XavierInitializer()));
    model.add(std::make_shared<Softmax>());

    // Compilar el modelo con pérdida y optimizador
    model.compile(
        std::make_shared<CrossEntropyLoss>(),
        std::make_shared<Adam>(0.001f)
    );

    // Cargar datos (ficticio, necesitarías implementarlo)
    Tensor train_images; // cargar imágenes MNIST como tensores
    Tensor train_labels; // etiquetas one-hot

    // Entrenar
    model.fit(train_images, train_labels, 10, 64); // ✅


    return 0;
}
