#include <iostream>
//#include "neuronal.hpp"
#include "hola.hpp"


int main() {
    MNISTLoader train_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 5000);
    MNISTLoader test_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 200);

    int patch_size = 7;
    int embed_dim = 64;
    int num_patches = (28 / patch_size) * (28 / patch_size);
    ModelBuilder builder;
    auto model = builder
        .add_patch_embedding(1, patch_size, embed_dim)
        .add_positional_encoding(num_patches, embed_dim, false)
        .add_transformer_encoder(embed_dim, 1, 64, 0.0f, ActivationType::ReLU)
        .add_global_average_pooling()
        .add_dense(embed_dim, 10, ActivationType::Softmax, 0.0f)
        .build();

    model->add_metric(std::make_shared<Accuracy>());
    model->compile(std::make_shared<CrossEntropyLoss>(),
                   std::make_shared<Adam>(0.001f),
                   std::make_shared<Logger>());
    model->fit(train_data.images, train_data.labels,
               test_data.images, test_data.labels,
               10, 32);

    return 0;
}
/*

auto model = builder
    .add_conv_block(1, 64, 3, 1, 1)            // Extraer features bajos (tipo CNN)
    .add_conv_block(64, 128, 3, 1, 1)
    .add_batchnorm1d(128)
    .add_activation(ActivationType::ReLU)
    .add_tokenizer(128, 16)                    // 16 tokens semánticos (tokenizer del paper)
    .add_transformer_token_block(128, 1, 128)  // VT encoder (self-attn + FFN)
    .add_projector(128, 128)                   // Reproyección al feature map (opcional si no haces segmentación)
    .add_global_average_pooling()
    .add_dense(128, 10, ActivationType::Softmax)
    .build();


    
int main() {
    MNISTLoader train_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1000);
    MNISTLoader test_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 200);

    int input_dim = 28 * 28;
    int hidden_dim = 128;
    int output_dim = 10;

    ModelBuilder builder;
    auto model = builder
        .add_flatten()
        .add_dense(input_dim, hidden_dim, ActivationType::ReLU, 0.0f)
        .add_dense(hidden_dim, output_dim, ActivationType::Softmax, 0.0f)
        .build();

    model->add_metric(std::make_shared<Accuracy>());
    model->compile(std::make_shared<CrossEntropyLoss>(),
                   std::make_shared<Adam>(0.001f),
                   std::make_shared<Logger>());
    model->fit(train_data.images, train_data.labels,
               test_data.images, test_data.labels,
               10, 32);

    return 0;
}



int main() {
    // Load MNIST training dataset with 1000 samples
    MNISTLoader train_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1000);

    // Load MNIST test dataset with 1000 samples
    MNISTLoader test_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 1000);

    // Create CNN model
    Model model;
    model.add(std::make_shared<Convolution>(1, 16, 3, 1, 1, std::make_shared<ReLU>(), 0.001f)); // 28x28x1 -> 28x28x16
    model.add(std::make_shared<MaxPooling>(2, 2)); // 28x28x16 -> 14x14x16
    model.add(std::make_shared<Dropout>(0.3f)); // Dropout 30%
    model.add(std::make_shared<Convolution>(16, 32, 3, 1, 1, std::make_shared<ReLU>(), 0.001f)); // 14x14x16 -> 14x14x32
    model.add(std::make_shared<MaxPooling>(2, 2)); // 14x14x32 -> 7x7x32
    model.add(std::make_shared<Flatten>()); // 7x7x32 -> 1568
    model.add(std::make_shared<Dense>(1568, 128, std::make_shared<ReLU>(), 0.001f)); // 1568 -> 128
    model.add(std::make_shared<Dropout>(0.5f)); // Dropout 50%
    model.add(std::make_shared<Dense>(128, 10, std::make_shared<Softmax>(), 0.0f)); // 128 -> 10, no weight decay

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
}*/