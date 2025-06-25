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
    model.add(std::make_shared<Conv2D>(1, 2, 3, 3, 1, "valid"));
    model.add(std::make_shared<ReLU>());
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
    model2.add(std::make_shared<ReLU>());
    model2.add(std::make_shared<AveragePooling2D>(2, 2, 2, "valid"));
    model2.add(std::make_shared<Flatten>());

    std::cout << "\n🔬 Iniciando demostración paso a paso...\n";
    model2.debug_pipeline_demo(input);

    return 0;
}
