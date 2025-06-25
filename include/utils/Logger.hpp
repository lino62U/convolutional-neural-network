#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

class Logger {
private:
    std::ofstream log_file;
    bool header_written = false;

public:
    Logger(const std::string& filename) {
        log_file.open(filename, std::ios::out);
        if (!log_file.is_open()) {
            throw std::runtime_error("No se pudo abrir el archivo de log.");
        }
    }

    void log_epoch(int epoch, float loss, float acc, float val_loss = -1.0f, float val_acc = -1.0f) {
        if (!header_written) {
            log_file << "epoch,train_loss,train_accuracy";
            if (val_loss >= 0 && val_acc >= 0)
                log_file << ",val_loss,val_accuracy";
            log_file << "\n";
            header_written = true;
        }

        log_file << epoch << ","
                 << std::fixed << std::setprecision(6) << loss << ","
                 << acc;
        if (val_loss >= 0 && val_acc >= 0) {
            log_file << "," << val_loss << "," << val_acc;
        }
        log_file << "\n";

        std::cout << "Epoch " << epoch
                  << " - Loss: " << loss
                  << " - Accuracy: " << acc;
        if (val_loss >= 0 && val_acc >= 0) {
            std::cout << " - Val Loss: " << val_loss
                      << " - Val Accuracy: " << val_acc;
        }
        std::cout << std::endl;
    }

    void info(const std::string& message) {
        log_file << "# " << message << std::endl;
        std::cout << "[INFO] " << message << std::endl;
    }

    ~Logger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }
};
