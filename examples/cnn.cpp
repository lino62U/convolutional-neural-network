#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <string>
#include <iomanip>

using namespace std;

struct TensorND {
    vector<float> data;
    vector<int> shape;

    int size() const {
        return accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    }

    int index(const vector<int>& idx) const {
        int i = 0, stride = 1;
        for (int d = shape.size() - 1; d >= 0; --d) {
            i += idx[d] * stride;
            stride *= shape[d];
        }
        return i;
    }

    float& at(const vector<int>& idx) {
        return data[index(idx)];
    }

    float at(const vector<int>& idx) const {
        return data[index(idx)];
    }

    void fillRandom(int min_val = 0, int max_val = 9) {
        for (auto& v : data)
            v = static_cast<float>(rand() % (max_val - min_val + 1) + min_val);
    }

    void print(const string& name) const {
        cout << "\n--- " << name << " (shape: ";
        for (int s : shape) cout << s << " ";
        cout << ") ---\n";

        vector<int> idx(shape.size(), 0);
        int count = 0;
        while (true) {
            cout << "[";
            for (size_t i = 0; i < idx.size(); ++i)
                cout << idx[i] << (i == idx.size() - 1 ? "] = " : ", ");
            cout << fixed << setprecision(2) << at(idx) << "\n";

            int d = idx.size() - 1;
            while (d >= 0 && ++idx[d] >= shape[d]) {
                idx[d] = 0;
                --d;
            }
            if (d < 0) break;
            if (++count > 1000) break;
        }
    }
};

TensorND applyPadding(const TensorND& input, const vector<int>& pad) {
    vector<int> new_shape(input.shape.size());
    for (size_t i = 0; i < input.shape.size(); ++i)
        new_shape[i] = input.shape[i] + 2 * pad[i];

    TensorND padded{vector<float>(accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>())), new_shape};

    vector<int> idx(input.shape.size(), 0);
    while (true) {
        vector<int> padded_idx = idx;
        for (size_t i = 0; i < padded_idx.size(); ++i)
            padded_idx[i] += pad[i];

        padded.at(padded_idx) = input.at(idx);

        int d = idx.size() - 1;
        while (d >= 0 && ++idx[d] >= input.shape[d]) {
            idx[d] = 0;
            --d;
        }
        if (d < 0) break;
    }

    return padded;
}

void convolveNDRecursive(
    const TensorND& input,
    const TensorND& kernel,
    TensorND& output,
    const vector<int>& stride,
    vector<int>& input_idx,
    vector<int>& output_idx,
    int dim
) {
    if (dim == input.shape.size()) {
        float val = 0.0f;
        vector<int> kernel_idx(kernel.shape.size(), 0);

        while (true) {
            vector<int> pos(input.shape.size());
            for (int i = 0; i < input.shape.size(); ++i)
                pos[i] = input_idx[i] + kernel_idx[i];

            val += input.at(pos) * kernel.at(kernel_idx);

            int d = kernel.shape.size() - 1;
            while (d >= 0 && ++kernel_idx[d] >= kernel.shape[d]) {
                kernel_idx[d] = 0;
                --d;
            }
            if (d < 0) break;
        }

        output.at(output_idx) = val;
        return;
    }

    for (int i = 0; i < output.shape[dim]; ++i) {
        input_idx[dim] = i * stride[dim];
        output_idx[dim] = i;
        convolveNDRecursive(input, kernel, output, stride, input_idx, output_idx, dim + 1);
    }
}

TensorND convolveND(const TensorND& input, const TensorND& kernel, const vector<int>& stride) {
    vector<int> out_shape(input.shape.size());
    for (size_t i = 0; i < input.shape.size(); ++i)
        out_shape[i] = (input.shape[i] - kernel.shape[i]) / stride[i] + 1;

    TensorND output{vector<float>(accumulate(out_shape.begin(), out_shape.end(), 1, multiplies<int>())), out_shape};

    vector<int> input_idx(input.shape.size(), 0);
    vector<int> output_idx(input.shape.size(), 0);
    convolveNDRecursive(input, kernel, output, stride, input_idx, output_idx, 0);

    return output;
}

/*
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Uso: " << argv[0] << " <dim> <num_filters> <padding valid|same> <input_shape...> <kernel_shape...> <stride...>\n";
        return 1;
    }

    int idx = 1;
    int dim = stoi(argv[idx++]);
    int num_filters = stoi(argv[idx++]);
    string padding_mode = argv[idx++];

    if (padding_mode != "valid" && padding_mode != "same") {
        cerr << "Error: padding debe ser 'valid' o 'same'.\n";
        return 1;
    }

    int expected_args = 1 + 3 + dim * 3;
    if (argc != expected_args) {
        cout << "Error: se esperaban " << expected_args << " argumentos, pero se recibieron " << argc << ".\n";
        return 1;
    }

    vector<int> input_shape(dim), kernel_shape(dim), stride(dim);
    for (int i = 0; i < dim; ++i) input_shape[i] = stoi(argv[idx++]);
    for (int i = 0; i < dim; ++i) kernel_shape[i] = stoi(argv[idx++]);
    for (int i = 0; i < dim; ++i) stride[i] = stoi(argv[idx++]);

    // Utilidad para imprimir vector como 3x3x3
    auto vec_to_str = [](const vector<int>& vec) {
        string s;
        for (size_t i = 0; i < vec.size(); ++i) {
            s += to_string(vec[i]);
            if (i != vec.size() - 1) s += "x";
        }
        return s;
    };

    // Información general
    cout << "======== CONFIGURACIÓN DE LA CONVOLUCIÓN ========\n";
    cout << "Dimensiones: " << dim << "\n";
    cout << "Tamaño de entrada: " << vec_to_str(input_shape) << "\n";
    cout << "Tamaño del kernel: " << vec_to_str(kernel_shape) << "\n";
    cout << "Stride: " << vec_to_str(stride) << "\n";
    cout << "Padding: " << padding_mode << "\n";
    cout << "Número de filtros: " << num_filters << "\n";
    cout << "=================================================\n";

    srand(time(0));

    TensorND input{vector<float>(accumulate(input_shape.begin(), input_shape.end(), 1, multiplies<int>())), input_shape};
    input.fillRandom();

    vector<TensorND> kernels;
    for (int f = 0; f < num_filters; ++f) {
        TensorND k{vector<float>(accumulate(kernel_shape.begin(), kernel_shape.end(), 1, multiplies<int>())), kernel_shape};
        k.fillRandom(-2, 2);
        kernels.push_back(k);
    }

    TensorND padded_input = input;
    if (padding_mode == "same") {
        vector<int> pad(dim);
        for (int i = 0; i < dim; ++i) {
            int total_pad = max(kernel_shape[i] - 1, 0);
            pad[i] = total_pad / 2;
        }
        padded_input = applyPadding(input, pad);
    }

    input.print("Tensor de Entrada");

    vector<TensorND> outputs;
    for (int f = 0; f < num_filters; ++f) {
        kernels[f].print("Kernel #" + to_string(f + 1));
        TensorND out = convolveND(padded_input, kernels[f], stride);
        out.print("Salida #" + to_string(f + 1));
        outputs.push_back(out);
    }

    return 0;
}
*/