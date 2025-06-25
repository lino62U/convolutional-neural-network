#pragma once

// Core
#include "core/Tensor.hpp"
#include "core/Layer.hpp"
#include "core/Model.hpp"
#include "core/Loss.hpp"
#include "core/Initializer.hpp"

// Activations
#include "activations/Activation.hpp"
#include "activations/ReLU.hpp"
#include "activations/Sigmoid.hpp"
#include "activations/Softmax.hpp"

// Layers
#include "layers/Dense.hpp"
#include "layers/Conv2D.hpp"
#include "layers/Flatten.hpp"
#include "layers/MaxPooling2D.hpp"
#include "layers/MinPooling2D.hpp"
#include "layers/AveragePooling2D.hpp"
#include "layers/Dropout.hpp"

// Puedes agregar más: Flatten, LSTM, RNN, etc.

// Optimizers
#include "optimizers/Optimizer.hpp"
#include "optimizers/SGD.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/RMSProp.hpp"

// Utils (si los necesitas)
#include "utils/DatasetLoader.hpp"
#include "utils/Logger.hpp"


// include/metrics/Metric.hpp
#include "metrics/Metric.hpp"
