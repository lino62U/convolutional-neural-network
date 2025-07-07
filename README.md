
# convolutional-neural-network

## Overview

This repository contains an implementation of a Convolutional Neural Network (CNN) designed for image classification tasks. The code is structured to provide modularity and ease of use, allowing users to experiment with different architectures and configurations.

## How to Execute

To run the project, execute the `run.sh` script located in the root directory. This script automates the setup and execution of the neural network training process. Ensure that the necessary dependencies are installed prior to execution.

### Steps:
1. Navigate to the project directory:
    ```bash
    cd /home/lupo/Documents/topicos_ia/convolutional-neural-network
    ```
2. Execute the script:
    ```bash
    ./run.sh
    ```

The script will initialize the training process, load the dataset, and begin training the CNN model.

## Includes

The project consists of the following key components:
- **Dataset Loader**: Handles the preprocessing and loading of image datasets.
- **Model Definitions**: Contains the implementation of CNN and MLP layers.
- **Training Module**: Manages the training loop, loss calculation, and optimization.
- **Utilities**: Includes helper functions for logging, visualization, and evaluation.

## Example Code

### Multi-Layer Perceptron (MLP) Layer
```cpp
#include <vector>
#include <cmath>

class MLP {
public:
     MLP(int input_size, int hidden_size, int output_size) {
          // Initialize weights and biases
          weights_input_hidden.resize(input_size * hidden_size);
          weights_hidden_output.resize(hidden_size * output_size);
          biases_hidden.resize(hidden_size);
          biases_output.resize(output_size);
     }

     std::vector<float> forward(const std::vector<float>& input) {
          // Compute hidden layer activations
          std::vector<float> hidden(hidden_size);
          for (int i = 0; i < hidden_size; ++i) {
                hidden[i] = biases_hidden[i];
                for (int j = 0; j < input_size; ++j) {
                     hidden[i] += input[j] * weights_input_hidden[j * hidden_size + i];
                }
                hidden[i] = std::tanh(hidden[i]); // Activation function
          }

          // Compute output layer activations
          std::vector<float> output(output_size);
          for (int i = 0; i < output_size; ++i) {
                output[i] = biases_output[i];
                for (int j = 0; j < hidden_size; ++j) {
                     output[i] += hidden[j] * weights_hidden_output[j * output_size + i];
                }
          }
          return output;
     }

private:
     int input_size, hidden_size, output_size;
     std::vector<float> weights_input_hidden, weights_hidden_output;
     std::vector<float> biases_hidden, biases_output;
};
```

### Convolutional Neural Network (CNN) Layer
```cpp
#include <vector>
#include <cmath>

class ConvLayer {
public:
     ConvLayer(int num_filters, int filter_size, int input_size) {
          this->num_filters = num_filters;
          this->filter_size = filter_size;
          this->input_size = input_size;

          // Initialize filters
          filters.resize(num_filters, std::vector<std::vector<float>>(filter_size, std::vector<float>(filter_size)));
     }

     std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input) {
          std::vector<std::vector<float>> output(input_size - filter_size + 1, std::vector<float>(input_size - filter_size + 1));

          for (int f = 0; f < num_filters; ++f) {
                for (int i = 0; i < input_size - filter_size + 1; ++i) {
                     for (int j = 0; j < input_size - filter_size + 1; ++j) {
                          float sum = 0.0f;
                          for (int k = 0; k < filter_size; ++k) {
                                for (int l = 0; l < filter_size; ++l) {
                                     sum += input[i + k][j + l] * filters[f][k][l];
                                }
                          }
                          output[i][j] = sum;
                     }
                }
          }
          return output;
     }

private:
     int num_filters, filter_size, input_size;
     std::vector<std::vector<std::vector<float>>> filters;
};
```

## Notes

- Ensure that the dataset is properly formatted and placed in the expected directory before running the script.
- Modify the configuration files to experiment with different architectures and hyperparameters.
## Layer Creation Example

The following examples demonstrate how to create layers for a dense network (MLP) and a Convolutional Neural Network (CNN) using the `src/main` file.

### Dense Network (MLP) Layer Creation
```cpp
#include "MLP.h"

int main() {
    // Define layer sizes
    int input_size = 128;
    int hidden_size = 64;
    int output_size = 10;

    // Create an MLP layer
    MLP dense_layer(input_size, hidden_size, output_size);

    // Example input
    std::vector<float> input(input_size, 1.0f);

    // Forward pass
    std::vector<float> output = dense_layer.forward(input);

    // Print output
    for (float value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### Convolutional Neural Network (CNN) Layer Creation
```cpp
#include "ConvLayer.h"

int main() {
    // Define layer parameters
    int num_filters = 3;
    int filter_size = 3;
    int input_size = 5;

    // Create a convolutional layer
    ConvLayer conv_layer(num_filters, filter_size, input_size);

    // Example input
    std::vector<std::vector<float>> input(input_size, std::vector<float>(input_size, 1.0f));

    // Forward pass
    std::vector<std::vector<float>> output = conv_layer.forward(input);

    // Print output
    for (const auto& row : output) {
        for (float value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

These examples illustrate how to instantiate and use the layers defined in the `src/main` file for building neural network architectures.