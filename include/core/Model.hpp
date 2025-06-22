// include/core/Model.hpp
#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include "Layer.hpp"
#include "Loss.hpp"
#include "optimizers/Optimizer.hpp"

class Model {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<Loss> loss;
    std::shared_ptr<Optimizer> optimizer;

public:
    void add(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    void compile(std::shared_ptr<Loss> loss_fn, std::shared_ptr<Optimizer> opt) {
        loss = loss_fn;
        optimizer = opt;
    }

    void fit(const Tensor& X, const Tensor& y, int epochs, int batch_size) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // --- FORWARD ---
            Tensor out = X;
            for (auto& layer : layers) {
                out = layer->forward(out);
            }

            float loss_value = loss->compute(out, y);
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << loss_value << std::endl;

            // --- BACKWARD ---
            Tensor grad = loss->gradient(out, y);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backward(grad);
            }

            // --- UPDATE ---
            for (auto& layer : layers) {
                layer->update_weights(optimizer.get());
            }
        }
    }

    Tensor predict(const Tensor& X) {
        Tensor out = X;
        for (auto& layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }
};
