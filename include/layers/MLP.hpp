#include <vector>
#include <memory>
#include "../core/Tensor.hpp"
#include "../core/Layer.hpp"

class MLP {
private:
    std::vector<std::shared_ptr<Layer>> layers;

public:
    void add_layer(const std::shared_ptr<Layer>& layer) {
        layers.push_back(layer);
    }

    Tensor forward(const Tensor& input) {
        Tensor output = input;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void backward(const Tensor& grad_output) {
        Tensor grad = grad_output;
        for (int i = layers.size() - 1; i >= 0; --i) {
            grad = layers[i]->backward(grad);
        }
    }

    void update(Optimizer* optimizer) {
        for (auto& layer : layers) {
            layer->update_weights(optimizer);
        }
    }
};
