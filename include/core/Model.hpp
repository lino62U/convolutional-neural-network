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

    void compile(std::shared_ptr<Loss> loss_fn, std::shared_ptr<Optimizer> opt,
                 std::shared_ptr<Logger> log = nullptr) {
        loss = loss_fn;
        optimizer = opt;
        logger = log;
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
            std::string log_message = "Evaluation - Loss: " + std::to_string(eval_loss);
            for (size_t m = 0; m < metrics.size(); ++m) {
                log_message += ", " + metrics[m]->name() + ": " + std::to_string(eval_metrics[m]);
            }
            logger->log(log_message);
        }
    }

    void fit(const Tensor& X, const Tensor& y, const Tensor& X_test, const Tensor& y_test, int epochs, int batch_size = 32) {
        if (X.shape[0] != y.shape[0]) {
            throw std::runtime_error("Input and target shape mismatch in training");
        }
        if (X_test.shape[0] != y_test.shape[0]) {
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
            }

            // Average metrics over batches
            epoch_loss /= num_batches;
            for (float& m : epoch_metrics) {
                m /= num_batches;
            }

            // Log training results
            if (logger) {
                std::string log_message = "Epoch " + std::to_string(epoch + 1) + "/" + 
                                         std::to_string(epochs) + ", Train Loss: " + 
                                         std::to_string(epoch_loss);
                for (size_t m = 0; m < metrics.size(); ++m) {
                    log_message += ", Train " + metrics[m]->name() + ": " + 
                                  std::to_string(epoch_metrics[m]);
                }
                logger->log(log_message);
            }

            // Evaluate on test set
            evaluate(X_test, y_test, batch_size);
        }
    }


    void print_tensor_shape(const Tensor& tensor) const {
        std::cout << "🔢 Shape: (";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            std::cout << tensor.shape[i];
            if (i < tensor.shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << ")\n";
    }

    void print_tensor_matrix(const Tensor& tensor) const {
        const auto& data = tensor.data;
        const auto& shape = tensor.shape;

        if (shape.size() == 4) {
            int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    std::cout << "🖼️ Sample " << n << ", canal " << c << ":\n";
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = n * C * H * W + c * H * W + h * W + w;
                            std::cout << data[idx] << "\t";
                        }
                        std::cout << "\n";
                    }
                }
            }
        } else if (shape.size() == 2) {
            int N = shape[0], F = shape[1];
            for (int n = 0; n < N; ++n) {
                std::cout << "🧾 Sample " << n << " (Flatten): ";
                for (int f = 0; f < F; ++f) {
                    int idx = n * F + f;
                    std::cout << data[idx] << " ";
                }
                std::cout << "\n";
            }
        } else if (shape.size() == 1) {
            std::cout << "📤 Vector plano: ";
            for (int i = 0; i < shape[0]; ++i) {
                std::cout << data[i] << " ";
            }
            std::cout << "\n";
        } else {
            std::cout << "⚠️  No se soporta impresión para tensores con " << shape.size() << " dimensiones.\n";
        }
    }




     void debug_pipeline_demo(const Tensor& input) {
        Tensor current = input;

        std::cout << "📥 Entrada:\n";
        current.print_shape();
        current.print_matrix();

        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "\n➡️ Paso por capa " << i << ": " << typeid(*layers[i]).name() << "\n";
            current = layers[i]->forward(current);
            current.print_shape();
            current.print_matrix();
        }

        std::cout << "\n✅ Resultado final\n";
        current.print_shape();
        current.print_matrix();
    }



    void fit2(const Tensor& X, const Tensor& y, int epochs, int batch_size,
            const Tensor* X_val = nullptr, const Tensor* y_val = nullptr) {

        int num_samples = X.shape[0];

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "\n🔁 Epoch " << (epoch + 1) << "/" << epochs << std::endl;
            float total_loss = 0.0f;
            std::vector<float> total_metrics(metrics.size(), 0.0f);
            int batches = 0;

            for (int i = 0; i < num_samples; i += batch_size) {
                std::cout << "  📦 Procesando batch desde índice " << i << std::endl;

                int end = std::min(i + batch_size, num_samples);
                Tensor X_batch = X.slice(i, end);
                Tensor y_batch = y.slice(i, end);

                // --- Mostrar input ---
                std::cout << "    🔎 Input X_batch:\n";
                print_tensor_shape(X_batch);
                print_tensor_matrix(X_batch);

                // --- Forward ---
                std::cout << "    ➡️  Forward..." << std::endl;
                Tensor out = X_batch;
                for (size_t l = 0; l < layers.size(); ++l) {
                    std::cout << "      🔹 Layer " << l << ": " << typeid(*layers[l]).name() << std::endl;
                    out = layers[l]->forward(out);
                }
                std::cout << "    ✅ Forward terminado" << std::endl;

                std::cout << "    🧮 Output del modelo:\n";
                print_tensor_shape(out);
                print_tensor_matrix(out);

                // --- Loss ---
                std::cout << "    📉 Calculando pérdida..." << std::endl;
                float loss_value = loss->compute(out, y_batch);
                total_loss += loss_value;
                std::cout << "    ✅ Pérdida: " << loss_value << std::endl;

                // --- Métricas ---
                for (size_t m = 0; m < metrics.size(); ++m) {
                    float metric_val = metrics[m]->compute(y_batch, out);
                    total_metrics[m] += metric_val;
                    std::cout << "    📊 Métrica[" << m << "]: " << metric_val << std::endl;
                }

                batches++;

                // --- Backward ---
                std::cout << "    🔁 Backward..." << std::endl;
                Tensor grad = loss->gradient(out, y_batch);
                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    grad = (*it)->backward(grad);
                }
                std::cout << "    ✅ Backward terminado" << std::endl;

                // --- Update ---
                std::cout << "    🛠️  Actualizando pesos..." << std::endl;
                for (auto& layer : layers) {
                    layer->update_weights(optimizer.get());
                }
                std::cout << "    ✅ Pesos actualizados" << std::endl;
            }

            float avg_loss = total_loss / batches;
            std::vector<float> avg_metrics;
            for (float val : total_metrics)
                avg_metrics.push_back(val / batches);

            // --- Validación ---
            float val_loss = -1.0f;
            std::vector<float> val_metrics(metrics.size(), -1.0f);

            if (X_val && y_val) {
                std::cout << "  🧪 Evaluando en datos de validación..." << std::endl;
                Tensor val_out = *X_val;
                for (auto& layer : layers)
                    val_out = layer->forward(val_out);

                std::cout << "  🔍 Output de validación:\n";
                print_tensor_shape(val_out);
                print_tensor_matrix(val_out);

                val_loss = loss->compute(val_out, *y_val);
                for (size_t m = 0; m < metrics.size(); ++m)
                    val_metrics[m] = metrics[m]->compute(*y_val, val_out);

                std::cout << "  ✅ Val loss: " << val_loss << std::endl;
                for (size_t m = 0; m < metrics.size(); ++m)
                    std::cout << "  📊 Val Métrica[" << m << "]: " << val_metrics[m] << std::endl;
            }

            // --- Logging final de la época ---
            if (logger) {
                std::cout << "📈 Epoch " << epoch + 1 << "/" << epochs
                        << " - Loss: " << avg_loss;
                for (size_t m = 0; m < metrics.size(); ++m)
                    std::cout << " - " << typeid(*metrics[m]).name() << ": " << avg_metrics[m];
                if (X_val && y_val) {
                    std::cout << " - Val Loss: " << val_loss;
                    for (size_t m = 0; m < metrics.size(); ++m)
                        std::cout << " - Val " << typeid(*metrics[m]).name() << ": " << val_metrics[m];
                }
                std::cout << std::endl;
            } else {
                std::cout << "📈 Epoch " << epoch + 1 << "/" << epochs
                        << " - Loss: " << avg_loss;
                for (size_t m = 0; m < metrics.size(); ++m)
                    std::cout << " - " << typeid(*metrics[m]).name() << ": " << avg_metrics[m];
                if (X_val && y_val) {
                    std::cout << " - Val Loss: " << val_loss;
                    for (size_t m = 0; m < metrics.size(); ++m)
                        std::cout << " - Val " << typeid(*metrics[m]).name() << ": " << val_metrics[m];
                }
                std::cout << std::endl;
            }
        }
    }



};


   