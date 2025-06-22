#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;
typedef vector<vector<float>> Matrix;

void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (float val : row)
            cout << setw(7) << fixed << setprecision(2) << val << " ";
        cout << endl;
    }
    cout << endl;
}

Matrix applyPadding(const Matrix& input, int pad) {
    int h = input.size();
    int w = input[0].size();
    Matrix padded(h + 2 * pad, vector<float>(w + 2 * pad, 0));
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            padded[i + pad][j + pad] = input[i][j];
    return padded;
}

Matrix convolve2D(const Matrix& input, const Matrix& kernel, int stride = 1, string padding_type = "valid") {
    int h = input.size(), w = input[0].size();
    int kh = kernel.size(), kw = kernel[0].size();
    int pad = 0;

    if (padding_type == "valid") {
        pad = (kh - 1) / 2;
    }

    Matrix padded = applyPadding(input, pad);
    int out_h = ((h + 2 * pad - kh) / stride) + 1;
    int out_w = ((w + 2 * pad - kw) / stride) + 1;
    Matrix output(out_h, vector<float>(out_w, 0.0));

    for (int i = 0; i < out_h; ++i)
        for (int j = 0; j < out_w; ++j)
            for (int m = 0; m < kh; ++m)
                for (int n = 0; n < kw; ++n)
                    output[i][j] += kernel[m][n] * padded[i * stride + m][j * stride + n];

    return output;
}

Matrix generateRandomKernel(int kh, int kw) {
    Matrix kernel(kh, vector<float>(kw));
    for (int i = 0; i < kh; ++i)
        for (int j = 0; j < kw; ++j)
            kernel[i][j] = static_cast<float>(rand() % 5 - 2); // entre -2 y 2
    return kernel;
}

Matrix generateRandomInput(int h, int w) {
    Matrix input(h, vector<float>(w));
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            input[i][j] = static_cast<float>(rand() % 10); // entre 0 y 9
    return input;
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        cout << "Uso: " << argv[0] << " <num_filtros> <kernel_size> <stride> <padding_type: valid|none> <input_height> <input_width>\n";
        return 1;
    }

    int num_filters = stoi(argv[1]);
    int kernel_size = stoi(argv[2]);
    int stride = stoi(argv[3]);
    string padding = argv[4];
    int input_height = stoi(argv[5]);
    int input_width = stoi(argv[6]);

    srand(time(0)); // Inicializa aleatoriedad

    Matrix input = generateRandomInput(input_height, input_width);

    // Mostrar parámetros
    cout << "\n========= PARÁMETROS =========\n";
    cout << "N° de filtros : " << num_filters << endl;
    cout << "Tamaño kernel : " << kernel_size << "x" << kernel_size << endl;
    cout << "Stride        : " << stride << endl;
    cout << "Padding       : " << padding << endl;
    cout << "Entrada       : " << input_height << "x" << input_width << endl;
    cout << "==============================\n\n";

    cout << "----- Matriz de Entrada (" << input_height << "x" << input_width << ") -----\n";
    printMatrix(input);

    for (int f = 0; f < num_filters; ++f) {
        cout << "\n========== FILTRO #" << f + 1 << " ==========\n";
        Matrix kernel = generateRandomKernel(kernel_size, kernel_size);

        cout << "--- Kernel " << kernel_size << "x" << kernel_size << " ---\n";
        printMatrix(kernel);

        Matrix result = convolve2D(input, kernel, stride, padding);

        cout << "--- Resultado de la Convolución ---\n";
        printMatrix(result);
        cout << "====================================\n";
    }

    return 0;
}
