#include "NeuralNet.hpp"
#include <iostream>
#include <memory>



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


void train_val_split(const Tensor& X, const Tensor& y,
                     float val_ratio,
                     Tensor& X_train, Tensor& y_train,
                     Tensor& X_val, Tensor& y_val) {
    if (X.shape[0] != y.shape[0])
        throw std::runtime_error("Shape mismatch between X and y");

    int total = X.shape[0];
    int val_size = static_cast<int>(val_ratio * total);
    int train_size = total - val_size;

    std::vector<int> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

    auto extract = [](const Tensor& full, const std::vector<int>& idxs, int from, int to) {
        int sample_size = to - from;
        int per_sample = full.total_elements() / full.shape[0];
        std::vector<float> out_data;
        for (int i = from; i < to; ++i) {
            int idx = idxs[i];
            out_data.insert(out_data.end(),
                            full.data.begin() + idx * per_sample,
                            full.data.begin() + (idx + 1) * per_sample);
        }
        std::vector<int> shape = full.shape;
        shape[0] = sample_size;
        return Tensor(out_data, shape);
    };

    X_train = extract(X, indices, 0, train_size);
    y_train = extract(y, indices, 0, train_size);
    X_val   = extract(X, indices, train_size, total);
    y_val   = extract(y, indices, train_size, total);
}




int main() {
    // Load MNIST training dataset with 1000 samples
    MNISTLoader train_data("data/fashionmnist/train-images-idx3-ubyte",
                           "data/fashionmnist/train-labels-idx1-ubyte", 1000);

    // Load MNIST test dataset with 1000 samples
    MNISTLoader test_data("data/fashionmnist/t10k-images-idx3-ubyte",
                          "data/fashionmnist/t10k-labels-idx1-ubyte", 1000);

    // Replicamos el canal 3 veces para obtener imágenes 28x28x3
    replicate_channels(train_data.images, 3);
    replicate_channels(test_data.images, 3);

    // Dividir en entrenamiento y validación
    Tensor X_train, y_train, X_val, y_val;
    train_val_split(train_data.images, train_data.labels,
                    0.2f,  // 20% para validación
                    X_train, y_train, X_val, y_val);

    std::cout << "Train shape: ";
    for (int d : X_train.shape) std::cout << d << " ";
    std::cout << "\nValidation shape: ";
    for (int d : X_val.shape) std::cout << d << " ";
    std::cout << std::endl;

    std::cout << "🧠 Construyendo modelo Vision Transformer...\n";
    Model model;

    // 28×28×3 → 28×28×16
    model.add(std::make_shared<Conv2D>(
        /*in*/3, /*out*/16, /*k*/3, /*s*/1, /*p*/1, std::make_shared<ReLU>()));
    //model.add( std::make_shared<ReLUActivationLayer>() );
    
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

    // 3136 → 10 (número de clases)
    model.add(std::make_shared<Dense>(
        3136, 10, std::make_shared<Softmax>()));

    /* ---------------------- 1. Resumen del modelo ----------------- */
    std::cout << "Modelo:\n";
    std::cout << " - Parámetros: " << model.num_params() << "\n";
    std::cout << " - Capas:\n";
    for (const auto& layer : model.get_layers()) {
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
    model.fit(X_train, y_train,
              X_val,   y_val,
              /*épocas*/ 5,
              /*batch*/  32);

    /* ---------------------- 5. Evaluación final -------------------- */
    model.evaluate(test_data.images, test_data.labels, 1);
}
