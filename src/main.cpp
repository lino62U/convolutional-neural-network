#include "NeuralNet.hpp"

#include <iostream>
#include <memory>

int main() {
    std::cout << "🚀 Inicio del programa\n";

    // Cargar datos con límite para depuración
    Tensor train_X, train_y, test_X, test_y;
    std::cout << "📦 Cargando datos MNIST...\n";
    DataLoader::load_mnist_data(train_X, train_y, test_X, test_y, 1000, 200);
    std::cout << "✅ Datos cargados exitosamente\n";

    // Crear modelo CNN
    std::cout << "🧠 Creando modelo CNN...\n";
    Model model;
    auto init = std::make_shared<XavierInitializer>();

    // Conv2D sin padding ("valid"), reduce tamaño
    model.add(std::make_shared<Conv2D>(1, 8, 3, 3, 1, 1)); // salida: (8, 26, 26)
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<MaxPooling2D>(2, 2));       // salida: (8, 13, 13)

    model.add(std::make_shared<Conv2D>(8, 16, 3, 3, 1, 1)); // salida: (16, 11, 11)
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<MaxPooling2D>(2, 2));       // salida: (16, 5, 5)

    model.add(std::make_shared<Flatten>());
    model.add(std::make_shared<Dense>(16 * 5 * 5, 64, init.get())); // entrada ajustada
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<Dense>(64, 10, init.get()));
    model.add(std::make_shared<Softmax>());

    std::cout << "✅ Modelo creado\n";

    // Compilar modelo
    auto loss_fn = std::make_shared<CrossEntropyLoss>();
    auto optimizer = std::make_shared<Adam>(0.001f);
    auto accuracy = std::make_shared<Accuracy>();

    std::cout << "🔧 Compilando modelo...\n";
    model.compile(loss_fn, optimizer, {accuracy});
    std::cout << "✅ Compilado correctamente\n";

    // Entrenar modelo
    std::cout << "🏋️ Entrenando modelo...\n";
    model.fit2(train_X, train_y, 5, 128, &test_X, &test_y);  // solo 2 épocas para prueba
    std::cout << "✅ Entrenamiento terminado\n";

    // Evaluar modelo final
    std::cout << "📊 Evaluando modelo...\n";
    float final_acc = model.evaluate(test_X, test_y);
    std::cout << "🔍 Accuracy final en test: " << final_acc << std::endl;

    return 0;
}
