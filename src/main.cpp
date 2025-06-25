#include "NeuralNet.hpp"
#include <iostream>
#include <memory>
/*
int main() {
    std::cout << "🚀 Inicio del programa\n";
    

    Tensor input({1, 1, 6, 6});
    input.data = {
        1, 2, 3, 4, 5, 6,
        6, -5, 4, 3, 2, 1,
        1, -3, 5, 3, 1, 0,
        0, 2, 4, 2, 4, 2,
        9, -8, 7, -6, 5, 4,
        4, 5, 6, 7, 8, 9
    };

    Model model;
    model.add(std::make_shared<Conv2D>(1, 2, 3, 3, 1, 1, "valid"));
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<MinPooling2D>(2, 2));
    model.add(std::make_shared<Flatten>());

    std::cout << "\n🔬 Iniciando demostración paso a paso...\n";
    model.debug_pipeline_demo(input);

    return 0;
    // 5 con la 7, clasificacion binaria funcioin que almacene el label 5 y 7
    // generar tres modelos diferentes 
    // luego usar el predicct y con eso la matriz de confusion
    // relu
    // sgd
}
*/


int main() {
    

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

    // Crear red
    Model model;
    Initializer* init = new XavierInitializer();

    model.add(std::make_shared<Flatten>());
    model.add(std::make_shared<Dense>(784, 10, init));
    model.add(std::make_shared<Softmax>());

    // Compilar modelo
    auto loss = std::make_shared<CrossEntropyLoss>();
    auto optimizer = std::make_shared<Adam>(0.001f);
    auto acc = std::make_shared<Accuracy>();
    model.compile(loss, optimizer, {acc});

    // Entrenar modelo
    model.fit(train_X, train_y, 10, 16);  // 10 épocas, batch size 16

    // Evaluar
    //model.evaluate(X, y);

    return 0;
}
