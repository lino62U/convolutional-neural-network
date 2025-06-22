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

class Model {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<Loss> loss;
    std::shared_ptr<Optimizer> optimizer;
    std::vector<std::shared_ptr<Metric>> metrics;
    std::shared_ptr<Logger> logger;  // Nuevo

public:
    void add(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    void compile(std::shared_ptr<Loss> loss_fn, std::shared_ptr<Optimizer> opt,
                 std::vector<std::shared_ptr<Metric>> metric_list = {},
                 std::shared_ptr<Logger> log = nullptr) {
        loss = loss_fn;
        optimizer = opt;
        metrics = metric_list;
        logger = log;
    }

     void fit(const Tensor& X, const Tensor& y, int epochs, int batch_size,
             const Tensor* X_val = nullptr, const Tensor* y_val = nullptr) {

        int num_samples = X.shape[0];

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            std::vector<float> total_metrics(metrics.size(), 0.0f);
            int batches = 0;

            for (int i = 0; i < num_samples; i += batch_size) {
                int end = std::min(i + batch_size, num_samples);

                Tensor X_batch = X.slice(i, end);
                Tensor y_batch = y.slice(i, end);

                // --- Forward ---
                Tensor out = X_batch;
                for (auto& layer : layers)
                    out = layer->forward(out);

                float loss_value = loss->compute(out, y_batch);
                total_loss += loss_value;

                for (size_t m = 0; m < metrics.size(); ++m)
                    total_metrics[m] += metrics[m]->compute(y_batch, out);

                batches++;

                // --- Backward ---
                Tensor grad = loss->gradient(out, y_batch);
                for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                    grad = (*it)->backward(grad);

                // --- Update ---
                for (auto& layer : layers)
                    layer->update_weights(optimizer.get());
            }

            // Promedios de entrenamiento
            float avg_loss = total_loss / batches;
            std::vector<float> avg_metrics;
            for (float val : total_metrics)
                avg_metrics.push_back(val / batches);

            // Validación
            float val_loss = -1.0f;
            std::vector<float> val_metrics(metrics.size(), -1.0f);

            if (X_val && y_val) {
                Tensor val_out = *X_val;
                for (auto& layer : layers)
                    val_out = layer->forward(val_out);

                val_loss = loss->compute(val_out, *y_val);

                for (size_t m = 0; m < metrics.size(); ++m)
                    val_metrics[m] = metrics[m]->compute(*y_val, val_out);
            }

            // Logging
            if (logger) {
                logger->log_epoch(epoch + 1, avg_loss, avg_metrics.empty() ? -1.0f : avg_metrics[0],
                  val_loss, val_metrics.empty() ? -1.0f : val_metrics[0]);


            } else {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
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

            // --- Forward ---
            std::cout << "    ➡️  Forward..." << std::endl;
            Tensor out = X_batch;
            for (size_t l = 0; l < layers.size(); ++l) {
                std::cout << "      🔹 Layer " << l << ": " << typeid(*layers[l]).name() << std::endl;
                out = layers[l]->forward(out);
            }
            std::cout << "    ✅ Forward terminado" << std::endl;

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

            val_loss = loss->compute(val_out, *y_val);
            for (size_t m = 0; m < metrics.size(); ++m)
                val_metrics[m] = metrics[m]->compute(*y_val, val_out);

            std::cout << "  ✅ Val loss: " << val_loss << std::endl;
            for (size_t m = 0; m < metrics.size(); ++m)
                std::cout << "  📊 Val Métrica[" << m << "]: " << val_metrics[m] << std::endl;
        }

        // --- Logging final de la época ---
        if (logger) {
            logger->log_epoch(epoch + 1, avg_loss,
                              avg_metrics.empty() ? -1.0f : avg_metrics[0],
                              val_loss,
                              val_metrics.empty() ? -1.0f : val_metrics[0]);
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


    Tensor predict(const Tensor& X) {
        Tensor out = X;
        for (auto& layer : layers)
            out = layer->forward(out);
        return out;
    }

    float evaluate(const Tensor& X, const Tensor& y) {
        Tensor out = X;
        for (auto& layer : layers)
            out = layer->forward(out);

        float acc_total = 0.0f;
        for (auto& metric : metrics) {
            float acc = metric->compute(y, out);
            std::cout << typeid(*metric).name() << ": " << acc << std::endl;
            acc_total += acc;
        }
        return acc_total / metrics.size();
    }
};
