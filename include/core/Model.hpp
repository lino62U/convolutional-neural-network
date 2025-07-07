#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include "Layer.hpp"
#include "Loss.hpp"
#include "optimizers/Optimizer.hpp"
#include "metrics/Metric.hpp"
#include "utils/Logger.hpp"  // Asegúrate de tener esta clase implementada
#include <random>


// Model class
class Model {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<Loss> loss;
    std::shared_ptr<Optimizer> optimizer;
    std::vector<std::shared_ptr<Metric>> metrics;
    std::shared_ptr<Logger> logger;
    bool training_mode;

public:
    Model() : training_mode(false) {}

    size_t num_params() const {
        return std::accumulate(layers.begin(), layers.end(), size_t(0),
            [](size_t total, const std::shared_ptr<Layer>& layer) {
                return total + layer->num_params();
            });
    }

    void add(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    void add_metric(std::shared_ptr<Metric> metric) {
        metrics.push_back(metric);
    }

    const std::vector<std::shared_ptr<Layer>>& get_layers() const {
        return layers;
    }

    void compile(std::shared_ptr<Loss> loss_fn, std::shared_ptr<Optimizer> opt,
                 std::shared_ptr<Logger> log = nullptr) {
        loss = loss_fn;
        optimizer = opt;
        logger = log;
    }

    void summary() const {
        std::cout << "🧠 Modelo resumen:\n";
        std::cout << " - Número total de parámetros: " << num_params() << "\n";
        std::cout << " - Capas (" << layers.size() << "):\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "   [" << i << "] " << typeid(*layers[i]).name() << "\n";
        }
    }


    Tensor forward(const Tensor& input, bool training = false) {
        Tensor current = input;
        for (auto& layer : layers) {
            current = layer->forward(current, training);
        }
        return current;
    }

    void evaluate(const Tensor& X, const Tensor& y, int batch_size = 32) {
        if (X.shape[0] != y.shape[0]) {
            throw std::runtime_error("Input and target shape mismatch in evaluation");
        }

        float eval_loss = 0.0f;
        std::vector<float> eval_metrics(metrics.size(), 0.0f);
        int num_batches = (X.shape[0] + batch_size - 1) / batch_size;

        training_mode = false; // Set to evaluation mode

        for (int b = 0; b < num_batches; ++b) {
            // Create batch
            int start = b * batch_size;
            int end = std::min(start + batch_size, X.shape[0]);
            int current_batch_size = end - start;
            
            // Extract batch data
            std::vector<float> batch_data;
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < X.total_elements() / X.shape[0]; ++j) {
                    batch_data.push_back(X.data[i * (X.total_elements() / X.shape[0]) + j]);
                }
            }
            std::vector<float> batch_target;
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < y.shape[1]; ++j) {
                    batch_target.push_back(y.data[i * y.shape[1] + j]);
                }
            }

            // Create batch tensors with correct shapes
            Tensor X_batch(batch_data, {current_batch_size, X.shape[1], X.shape[2], X.shape[3]});
            Tensor y_batch(batch_target, {current_batch_size, y.shape[1]});

            // Forward pass
            Tensor y_pred = forward(X_batch, false);
            // 🔥 Liberar caché
            for (auto& layer : layers) {
                layer->clear_cache();
            }


            // Compute loss
            eval_loss += loss->compute(y_pred, y_batch);

            // Compute metrics
            for (size_t m = 0; m < metrics.size(); ++m) {
                eval_metrics[m] += metrics[m]->compute(y_pred, y_batch);
            }
        }

        // Average metrics over batches
        eval_loss /= num_batches;
        for (float& m : eval_metrics) {
            m /= num_batches;
        }

        // Log evaluation results
        if (logger) {
            std::vector<std::pair<std::string, float>> val_metrics_vec;
            for (size_t m = 0; m < metrics.size(); ++m) {
                val_metrics_vec.emplace_back(metrics[m]->name(), eval_metrics[m]);
            }
            logger->log_eval(eval_loss, val_metrics_vec);
        }
    }

    void fit(const Tensor& X, const Tensor& y, const Tensor& X_val, const Tensor& y_val, int epochs, int batch_size = 32) {
        if (X.shape[0] != y.shape[0]) {
            throw std::runtime_error("Input and target shape mismatch in training");
        }
        if (X_val.shape[0] != y_val.shape[0]) {
            throw std::runtime_error("Input and target shape mismatch in test set");
        }

        std::vector<int> indices(X.shape[0]);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(std::random_device{}());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);
            
            float epoch_loss = 0.0f;
            std::vector<float> epoch_metrics(metrics.size(), 0.0f);
            int num_batches = (X.shape[0] + batch_size - 1) / batch_size;

            training_mode = true; // Set to training mode

//            std::cout << "\n📦 Época " << epoch + 1 << "/" << epochs << "\n";
            for (int b = 0; b < num_batches; ++b) {
                // Create batch
                int start = b * batch_size;
                int end = std::min(start + batch_size, X.shape[0]);
                int current_batch_size = end - start;
                
                // Extract batch data
                std::vector<float> batch_data;
                for (int i = start; i < end; ++i) {
                    int idx = indices[i];
                    for (int j = 0; j < X.total_elements() / X.shape[0]; ++j) {
                        batch_data.push_back(X.data[idx * (X.total_elements() / X.shape[0]) + j]);
                    }
                }
                std::vector<float> batch_target;
                for (int i = start; i < end; ++i) {
                    int idx = indices[i];
                    for (int j = 0; j < y.shape[1]; ++j) {
                        batch_target.push_back(y.data[idx * y.shape[1] + j]);
                    }
                }

                // Create batch tensors with correct shapes
                Tensor X_batch(batch_data, {current_batch_size, X.shape[1], X.shape[2], X.shape[3]});
                Tensor y_batch(batch_target, {current_batch_size, y.shape[1]});

                // Forward pass
                Tensor y_pred = forward(X_batch, true);

                // Compute loss
                float batch_loss = loss->compute(y_pred, y_batch);
                epoch_loss += batch_loss;

                // Compute metrics
                for (size_t m = 0; m < metrics.size(); ++m) {
                    epoch_metrics[m] += metrics[m]->compute(y_pred, y_batch);
                }

                // Backward pass
                Tensor grad = loss->gradient(y_pred, y_batch);
                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    grad = (*it)->backward(grad);
                }

                // Update weights
                for (auto& layer : layers) {
                    layer->update_weights(optimizer.get());
                }

                // 🔥 Clear caches después del backward y update
                for (auto& layer : layers) {
                    layer->clear_cache();
                }
            }

            // Average metrics over batches
            epoch_loss /= num_batches;
            for (float& m : epoch_metrics) {
                m /= num_batches;
            }

            // Evaluate on test set
            float eval_loss = 0.0f;
            std::vector<float> eval_metrics(metrics.size(), 0.0f);
            {
                training_mode = false;
                Tensor y_pred = forward(X_val, false);
                eval_loss = loss->compute(y_pred, y_val);

                for (size_t m = 0; m < metrics.size(); ++m) {
                    eval_metrics[m] = metrics[m]->compute(y_pred, y_val);
                }
            }
            // Log training results
            if (logger) {
                std::vector<std::pair<std::string, float>> train_metrics_vec;
                for (size_t m = 0; m < metrics.size(); ++m) {
                    train_metrics_vec.emplace_back(metrics[m]->name(), epoch_metrics[m]);
                }

                std::vector<std::pair<std::string, float>> val_metrics_vec;
                for (size_t m = 0; m < metrics.size(); ++m) {
                    val_metrics_vec.emplace_back(metrics[m]->name(), eval_metrics[m]);
                }

                logger->log_epoch(epoch + 1, epochs,
                                epoch_loss, train_metrics_vec,
                                eval_loss, val_metrics_vec);
            }

            // Evaluate on test set
            //evaluate(X_val, y_val, batch_size);
        }
    }




};


   