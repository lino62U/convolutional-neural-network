#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>

class Logger {
private:
    std::ofstream file;
    std::string log_path;
    bool header_written = false;

public:
    Logger(const std::string& path = "training_log.csv") : log_path(path) {
        auto dir = std::filesystem::path(path).parent_path();
        if (!dir.empty()) {
            std::filesystem::create_directories(dir);
        }

        file.open(log_path, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open log file: " + log_path);
        }
    }


    void log_epoch(int epoch, int total_epochs,
               float train_loss, const std::vector<std::pair<std::string, float>>& train_metrics,
               float val_loss, const std::vector<std::pair<std::string, float>>& val_metrics) {
        std::ostringstream oss;

        // --- Consola: formato elegante
        oss << "[Epoch " << epoch << "/" << total_epochs << "]"
            << " | Train Loss: " << std::fixed << std::setprecision(6) << train_loss;

        for (const auto& m : train_metrics) {
            oss << " | Train " << capitalize(m.first) << ": " << std::fixed << std::setprecision(6) << m.second;
        }

        oss << " | Val Loss: " << std::fixed << std::setprecision(6) << val_loss;

        for (const auto& m : val_metrics) {
            oss << " | Val " << capitalize(m.first) << ": " << std::fixed << std::setprecision(6) << m.second;
        }

        std::cout << oss.str() << std::endl;

        // --- CSV (igual que antes)
        if (!header_written) {
            file << "epoch,train_loss";
            for (const auto& m : train_metrics) file << ",train_" << m.first;
            file << ",val_loss";
            for (const auto& m : val_metrics) file << ",val_" << m.first;
            file << std::endl;
            header_written = true;
        }

        file << epoch << "," << train_loss;
        for (const auto& m : train_metrics) file << "," << m.second;
        file << "," << val_loss;
        for (const auto& m : val_metrics) file << "," << m.second;
        file << std::endl;
    }

    void log_eval(float eval_loss, const std::vector<std::pair<std::string, float>>& val_metrics) {
        std::ostringstream oss;
        oss << "[Evaluation] | Val Loss: " << std::fixed << std::setprecision(6) << eval_loss;
        for (const auto& m : val_metrics) {
            oss << " | Val " << capitalize(m.first) << ": " << std::fixed << std::setprecision(6) << m.second;
        }
        std::cout << oss.str() << std::endl;
    }

    std::string capitalize(const std::string& s) {
        if (s.empty()) return s;
        std::string out = s;
        out[0] = std::toupper(out[0]);
        return out;
    }

    ~Logger() {
        if (file.is_open()) file.close();
    }
};


