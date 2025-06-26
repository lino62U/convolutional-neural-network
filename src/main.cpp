#include "NeuralNet.hpp"
#include <iostream>
#include <memory>


/*
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
    model.add(std::make_shared<Conv2D>(1, 2, 3, 0, 1));
    model.add(std::make_shared<ReLUActivationLayer>()); // como capa
    model.add(std::make_shared<MaxPooling2D>(2, 2));   // solo (size, stride)
    model.add(std::make_shared<Flatten>());

    std::cout << "\n🔬 Demo paso a paso modelo 1\n";
    model.debug_pipeline_demo(input);

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
    model2.add(std::make_shared<Conv2D>(1, 2, 3, 0, 1, std::make_shared<ReLU>()));
    model2.add(std::make_shared<AveragePooling2D>(2, 2));
    model2.add(std::make_shared<Flatten>());

    std::cout << "\n🔬 Iniciando demostración paso a paso...\n";
    model2.debug_pipeline_demo(input);

    return 0;
}

*/
int main() {
    std::cout << "🔁 Cargando datos...\n";
    MNISTLoader train_data("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 1000);
    MNISTLoader test_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 1000);

    std::cout << "🧠 Construyendo modelo CNN...\n";
    Model model;

    // Entrada: (N, 1, 28, 28)
    model.add(std::make_shared<Conv2D>(1, 8, 3, PaddingType::VALID, 1));  // 28x28x1 -> 26x26x8 (kernel 3x3 reduce 2px en cada dimensión)
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<MaxPooling2D>(2, 2));          // 26x26x8 -> 13x13x8
    model.add(std::make_shared<Dropout>(0.25f));

    model.add(std::make_shared<Conv2D>(8, 16, 3, PaddingType::VALID, 1)); // 13x13x8 -> 11x11x16
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<MaxPooling2D>(2, 2));          // 11x11x16 -> 5x5x16 (redondeo hacia abajo)
    model.add(std::make_shared<Dropout>(0.25f));

    model.add(std::make_shared<Flatten>());                 // 5x5x16 = 400
    model.add(std::make_shared<Dense>(400, 64));            // 400 -> 64
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<Dropout>(0.5f));

    model.add(std::make_shared<Dense>(64, 10));             // 64 -> 10
    model.add(std::make_shared<Softmax>());


    std::cout << "⚙️  Configurando modelo...\n";
    model.add_metric(std::make_shared<Accuracy>());
    model.compile(
        std::make_shared<CrossEntropyLoss>(),
        std::make_shared<Adam>(0.001f),
        std::make_shared<Logger>()
    );

    std::cout << "🚀 Entrenando...\n";
    model.fit(train_data.images, train_data.labels,
              test_data.images, test_data.labels,
              10, 32);

    return 0;
}