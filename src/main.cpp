#include "NeuralNet.hpp"
#include <iostream>
#include <memory>



using namespace utils;

int main() {
    // Variables de inicialización
    int num_train_samples = 60000;
    int num_test_samples = 10000;
    float val_ratio = 0.2f;
    int num_epochs = 20;
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


    // Elegir modelo CNN 
    
    std::cout << "Construyendo modelo CNN...\n";
    Model cnn_model = build_cnn_model();

    // Mostrar resumen CNN
    cnn_model.summary();
    std::cout << "Modelo CNN:\n";

    cnn_model.add_metric(std::make_shared<Accuracy>());
    cnn_model.compile(
        std::make_shared<CrossEntropyLoss>(),
        std::make_shared<SGD>(learning_rate),
        std::make_shared<Logger>("training_log_cnn.csv"));

    cnn_model.fit(X_train, y_train, X_val, y_val, num_epochs, batch_size);
    
    std::cout << "Evaluación final CNN:\n";
    cnn_model.evaluate(test_data.images, test_data.labels, batch_size);

    // Elegir modelo MLP
    std::cout << "Construyendo modelo MLP...\n";
    Model mlp_model = build_mlp_model();


    // Mostrar resumen MLP
    mlp_model.summary();
    std::cout << "Modelo MLP:\n";

    mlp_model.add_metric(std::make_shared<Accuracy>());
    mlp_model.compile(
        std::make_shared<CrossEntropyLoss>(),
        std::make_shared<SGD>(learning_rate),
        std::make_shared<Logger>("training_log_mlp.csv"));

    mlp_model.fit(X_train, y_train, X_val, y_val, num_epochs, batch_size);
    
    std::cout << "Evaluación final MLP:\n";
    mlp_model.evaluate(test_data.images, test_data.labels, batch_size);
}
