#include "NeuralNet.hpp"
#include <iostream>
#include <memory>



using namespace utils;

int main() {
    // Variables de inicialización
    int num_train_samples = 1000;
    int num_test_samples = 1000;
    float val_ratio = 0.2f;
    int num_epochs = 5;
    int batch_size = 32;
    float learning_rate = 0.002f;

    // Load MNIST training dataset
    MNISTLoader train_data("data/fashionmnist/train-images-idx3-ubyte",
                           "data/fashionmnist/train-labels-idx1-ubyte", num_train_samples);
    MNISTLoader test_data("data/fashionmnist/t10k-images-idx3-ubyte",
                          "data/fashionmnist/t10k-labels-idx1-ubyte", num_test_samples);

    // Replicar a 3 canales
    replicate_channels(train_data.images, 3);
    replicate_channels(test_data.images, 3);

    // División train / val
    Tensor X_train, y_train, X_val, y_val;
    train_val_split(train_data.images, train_data.labels, val_ratio,
                    X_train, y_train, X_val, y_val);

    std::cout << "Train shape: ";
    for (int d : X_train.shape) std::cout << d << " ";
    std::cout << "\nValidation shape: ";
    for (int d : X_val.shape) std::cout << d << " ";
    std::cout << "\nTest shape: ";
    for (int d : test_data.images.shape) std::cout << d << " ";
    std::cout << std::endl;

    std::cout << " Construyendo modelo CNN...\n";
    Model model;

    // 28×28×3 → 28×28×16
    auto conv1 = std::make_shared<Conv2D>(
        3, 16, 3, PaddingType::CUSTOM, 1);
    conv1->set_custom_padding(1); // padding = 1
    model.add(conv1);
    model.add(std::make_shared<ReLU>());
    // 28×28×16 → 14×14×16
    model.add(std::make_shared<MaxPooling2D>(2, 2));

    // 14×14×16 → 14×14×64
    auto conv2 = std::make_shared<Conv2D>(
        16, 64, 3, PaddingType::CUSTOM, 1);
    conv2->set_custom_padding(1); // padding = 1
    model.add(conv2);

    model.add(std::make_shared<ReLU>());

    // 14×14×64 → 7×7×64
    model.add(std::make_shared<MaxPooling2D>(2, 2));

    // 7×7×64 → 3136
    model.add(std::make_shared<Flatten>());

    // 3136 → 10 (clases) (Capa densa)
    model.add(std::make_shared<Linear>(3136, 10));
    model.add(std::make_shared<Softmax>());

    // Resumen
    std::cout << "Modelo:\n";
    std::cout << " - Parámetros: " << model.num_params() << "\n";
    std::cout << " - Capas:\n";
    for (const auto& layer : model.get_layers())
        std::cout << "   - " << typeid(*layer).name() << "\n";

    // Métricas
    model.add_metric(std::make_shared<Accuracy>());

    // Compilar
    model.compile(
        std::make_shared<CrossEntropyLoss>(),
        std::make_shared<SGD>(learning_rate),
        std::make_shared<Logger>());

    // Entrenar
    model.fit(X_train, y_train, X_val, y_val, num_epochs, batch_size);

    // Evaluar
    model.evaluate(test_data.images, test_data.labels, batch_size);
}
