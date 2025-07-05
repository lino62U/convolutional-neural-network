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

void replicate_channels(Tensor& batch, int channels) {
    if (batch.shape.size() != 4 || batch.shape[1] != 1)
        throw std::runtime_error("Expected shape [N, 1, H, W]");

    int batch_size = batch.shape[0];
    int height = batch.shape[2];
    int width = batch.shape[3];
    int hw = height * width;

    std::vector<float> new_data(batch_size * channels * hw);

    for (int n = 0; n < batch_size; ++n) {
        const float* src = batch.data.data() + n * hw;
        for (int c = 0; c < channels; ++c) {
            float* dst = new_data.data() + (n * channels + c) * hw;
            std::copy(src, src + hw, dst);
        }
    }

    batch.data = std::move(new_data);
    batch.shape = {batch_size, channels, height, width};
}





int main() {
    // Load MNIST training dataset with 1000 samples
    MNISTLoader train_data("data/fashionmnist/train-images-idx3-ubyte", "data/fashionmnist/train-labels-idx1-ubyte", 1000);

    // Load MNIST test dataset with 1000 samples
    MNISTLoader test_data("data/fashionmnist/t10k-images-idx3-ubyte", "data/fashionmnist/t10k-labels-idx1-ubyte", 1000);

    // Replicamos el canal 3 veces para obtener imágenes 28x28x3
    replicate_channels(train_data.images, 3);
    replicate_channels(test_data.images, 3);

    std::cout << "Train shape: ";
    for (int d : train_data.images.shape) std::cout << d << " ";
    std::cout << std::endl;



    // Create CNN model
     Model model;

    // 28×28×1 → 28×28×16
    model.add(std::make_shared<Conv2D>(
        /*in*/3, /*out*/16, /*k*/3, /*s*/1, /*p*/1,
        std::make_shared<ReLU>()));
    // 28×28×16 → 14×14×16
    model.add(std::make_shared<MaxPooling2D>(2, 2));

    // 14×14×16 → 14×14×64
    model.add(std::make_shared<Conv2D>(
        /*in*/16, /*out*/64, /*k*/3, /*s*/1, /*p*/1,
        std::make_shared<ReLU>()));
    // 14×14×64 → 7×7×64
    model.add(std::make_shared<MaxPooling2D>(2, 2));

    // 7×7×64 → 3136
    model.add(std::make_shared<Flatten>());

    // 3136 → 16
    model.add(std::make_shared<Dense>(
        3136, 10, std::make_shared<Softmax>()));

    /* ---------------------- 1. Resumen del modelo ----------------- */
    std::cout << "Modelo:\n";
    std::cout << " - Parámetros: " << model.num_params() << "\n";
    std::cout << " - Capas:\n";
    for (const auto& layer : model.get_layers()) {
        // Usa typeid para mostrar el nombre de la clase de la capa
        std::cout << "   - " << typeid(*layer).name() << "\n";
    }   

    // Métrica
    model.add_metric(std::make_shared<Accuracy>());

    /* ---------------------- 3. Compilación ------------------------- */
    model.compile(
        std::make_shared<CrossEntropyLoss>(),
        std::make_shared<SGD>(0.002f),   // LR = 0.002
        std::make_shared<Logger>());

    /* ---------------------- 4. Entrenamiento ----------------------- */
    model.fit(train_data.images, train_data.labels,
              test_data.images,  test_data.labels,
              /*épocas*/20,
              /*batch*/1);
    /* ---------------------- 5. Evaluación ------------------------- */
}