#include "NeuralNet.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    try {
        std::cout << "🚀 Inicio del programa\n";

        // --- Carga de datos ---
        Tensor train_X, train_y, test_X, test_y;
        constexpr int train_samples = 1000;
        constexpr int test_samples = 200;

        std::cout << "📦 Cargando datos MNIST...\n";
        DataLoader::load_mnist_data(train_X, train_y, test_X, test_y, train_samples, test_samples);
        
        if (train_X.empty() || test_X.empty()) {
            throw std::runtime_error("Error: Datos MNIST no cargados");
        }
        std::cout << "✅ Datos cargados (" << train_X.shape[0] << " train, " 
                  << test_X.shape[0] << " test)\n";

        // --- Construcción del Modelo ---
        std::cout << "🧠 Creando modelo CNN...\n";
        Model model;
        auto init = std::make_unique<XavierInitializer>();

        // Capas convolucionales
        model.add(std::make_shared<Conv2D>(1, 8, 3, 3, 1, 1));  // Output: (8, 26, 26)
        model.add(std::make_shared<ReLU>());
        model.add(std::make_shared<MaxPooling2D>(2, 2));        // Output: (8, 13, 13)

        model.add(std::make_shared<Conv2D>(8, 16, 3, 3, 1, 1)); // Output: (16, 11, 11)
        model.add(std::make_shared<ReLU>());
        model.add(std::make_shared<MaxPooling2D>(2, 2));       // Output: (16, 5, 5)

        // Capas fully connected
        model.add(std::make_shared<Flatten>());
        model.add(std::make_shared<Dense>(16 * 5 * 5, 64, init.get()));
        model.add(std::make_shared<ReLU>());
        model.add(std::make_shared<Dense>(64, 10, init.get()));
        model.add(std::make_shared<Softmax>());

        std::cout << "✅ Modelo creado (" << model.num_params() << " parámetros)\n";

        // --- Compilación ---
        auto loss_fn = std::make_shared<CrossEntropyLoss>();
        auto optimizer = std::make_shared<Adam>(0.001f);
        auto accuracy = std::make_shared<Accuracy>();

        std::cout << "🔧 Compilando modelo...\n";
        model.compile(loss_fn, optimizer, {accuracy});
        std::cout << "✅ Modelo compilado\n";

        // --- Entrenamiento ---
        constexpr int epochs = 5;
        constexpr int batch_size = 128;

        std::cout << "🏋️ Entrenando modelo (" << epochs << " épocas)...\n";
        model.fit(train_X, train_y, epochs, batch_size, &test_X, &test_y);
        std::cout << "✅ Entrenamiento completado\n";

        // --- Evaluación Final ---
        std::cout << "📊 Evaluando modelo con test set...\n";
        float final_acc = model.evaluate(test_X, test_y);
        std::cout << "🔍 Accuracy final en test: " << final_acc * 100 << "%\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}