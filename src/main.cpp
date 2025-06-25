#include "NeuralNet.hpp"
#include <iostream>
#include <memory>


int main() {
    std::cout << "🚀 Inicio del programa\n";

    Tensor input({1, 1, 6, 6});
    input.data = {
        1, -5, -8, 4, 5, 6,
        6, -5, 4, 3, 2, 1,
        1, -3, 5, 3, 1, 0,
        0, 2, 4, 2, 4, 2,
        9, -8, 7, -6, 5, 4,
        4, 5, 6, 7, 8, 9
    };

    std::cout << "\n======================== Modelo 1 ========================\n";
    std::cout << "🔧 Configuración:\n";
    std::cout << " - Conv2D: 1 canal -> 2 canales, kernel=3x3, stride=1, padding='valid'\n";
    std::cout << " - Activación: ReLU\n";
    std::cout << " - Pooling: MaxPooling2D, tamaño=2x2, stride=2, padding='valid'\n";
    std::cout << " - Flatten\n";
    std::cout << "📝 Descripción: MaxPooling con stride 2 sin padding\n";

    Model model;
    model.add(std::make_shared<Conv2D>(16, 32, 3, 1, 1, std::make_shared<ReLU>())); // 14x14x16 -> 14x14x32
    //model.add(std::make_shared<Conv2D>(1, 2, 3, 3, 1, "valid"));

    model.add(std::make_shared<MaxPooling2D>(2, 2, 2, "valid"));
    model.add(std::make_shared<Flatten>());

    std::cout << "\n🔬 Iniciando demostración paso a paso...\n";
    model.debug_pipeline_demo(input);

    std::cout << "\n======================== Modelo 2 ========================\n";
    std::cout << "🔧 Configuración:\n";
    std::cout << " - Conv2D: 1 canal -> 2 canales, kernel=3x3, stride=1, padding='valid'\n";
    std::cout << " - Activación: ReLU\n";
    std::cout << " - Pooling: AveragePooling2D, tamaño=2x2, stride=2, padding='valid'\n";
    std::cout << " - Flatten\n";
    std::cout << "📝 Descripción: AveragePooling con stride 2 sin padding\n";

    Model model2;
    model2.add(std::make_shared<Conv2D>(1, 2, 3, 3, 1, "valid"));
 
    model2.add(std::make_shared<AveragePooling2D>(2, 2, 2, "valid"));
    model2.add(std::make_shared<Flatten>());

    std::cout << "\n🔬 Iniciando demostración paso a paso...\n";
    model2.debug_pipeline_demo(input);

    return 0;
}


/*
int main() {
    // Load MNIST training dataset with 1000 samples
    MNISTLoader train_data("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 1000);

    // Load MNIST test dataset with 1000 samples
    MNISTLoader test_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 1000);

    // Create CNN model
    Model model;
    model.add(std::make_shared<Conv2D>(1, 16, 3, 1, 1, std::make_shared<ReLU>())); // 28x28x1 -> 28x28x16
    model.add(std::make_shared<MaxPooling2D>(2, 2)); // 28x28x16 -> 14x14x16

    model.add(std::make_shared<Dropout>(0.3f)); // Dropout 30%


    model.add(std::make_shared<Conv2D>(16, 32, 3, 1, 1, std::make_shared<ReLU>())); // 14x14x16 -> 14x14x32
    model.add(std::make_shared<MaxPooling2D>(2, 2)); // 14x14x32 -> 7x7x32
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
*/