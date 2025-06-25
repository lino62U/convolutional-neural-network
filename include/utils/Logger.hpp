#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

// Basic Logger class
class Logger {
public:
    virtual void log(const std::string& message) {
        std::cout << message << std::endl;
    }
    virtual ~Logger() {}
};