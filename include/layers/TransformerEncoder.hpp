
#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include <memory>
#include <vector>

#include "core/Layer.hpp"

#include "activations/Activation.hpp"
#include "activations/ReLU.hpp"
#include "activations/Sigmoid.hpp"
#include "activations/Tanh.hpp"
#include "LayerNorm.hpp"
#include "MultiHeadSelfAttention.hpp"
#include "FeedForward.hpp"


class TransformerEncoder : public Layer {
private:
    std::shared_ptr<LayerNorm> ln1;
    std::shared_ptr<MultiHeadSelfAttention> mhsa;
    std::shared_ptr<LayerNorm> ln2;
    std::shared_ptr<FeedForward> ff;

    // Caches
    Tensor input_cache;
    Tensor post_mhsa_cache;

public:
    TransformerEncoder(int embed_dim, int num_heads, int ff_dim, float dropout_rate, ActivationType act_type)
        : ln1(std::make_shared<LayerNorm>(embed_dim)),
          mhsa(std::make_shared<MultiHeadSelfAttention>(embed_dim, num_heads, dropout_rate)),
          ln2(std::make_shared<LayerNorm>(embed_dim)),
          ff(std::make_shared<FeedForward>(embed_dim, ff_dim, act_type, dropout_rate)) {}

    Tensor forward(const Tensor& input, bool training = false) override {
        input_cache = input;  // Save for backward

        // Pre-LN variant
        Tensor x1 = ln1->forward(input, training);
        Tensor mhsa_out = mhsa->forward(x1, training);
        Tensor post_mhsa = input + mhsa_out;
        post_mhsa_cache = post_mhsa;  // Save for backward

        Tensor x2 = ln2->forward(post_mhsa, training);
        Tensor ff_out = ff->forward(x2, training);
        Tensor output = post_mhsa + ff_out;

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        // Step 1: backward through FF and residual
        Tensor grad_ff = ff->backward(grad_output);  // dL/d(x2)
        Tensor grad_ln2 = ln2->backward(grad_ff);    // dL/d(post_mhsa)

        Tensor grad_post_mhsa = grad_ln2 + grad_output;

        // Step 2: backward through MHSA and residual
        Tensor grad_mhsa = mhsa->backward(grad_post_mhsa);  // dL/d(x1)
        Tensor grad_ln1 = ln1->backward(grad_mhsa);         // dL/d(input_norm)

        // Step 3: skip connection (gradient of residual path is just grad_post_mhsa)
        Tensor grad_input = grad_ln1 + grad_post_mhsa;  // ∂L/∂input from both branches

        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        ln1->update_weights(optimizer);
        mhsa->update_weights(optimizer);
        ln2->update_weights(optimizer);
        ff->update_weights(optimizer);
    }

    size_t num_params() const override {
        return ln1->num_params() + mhsa->num_params() + ln2->num_params() + ff->num_params();
    }
};

    /*
class TransformerEncoder : public Layer {
private:
    std::shared_ptr<LayerNorm> ln1;
    std::shared_ptr<MultiHeadSelfAttention> mhsa;
    std::shared_ptr<LayerNorm> ln2;
    std::shared_ptr<FeedForward> ff;
    Tensor mhsa_cache;
    Tensor ff_cache;

public:
    TransformerEncoder(int embed_dim, int num_heads, int ff_dim, float dropout_rate, ActivationType act_type)
        : ln1(std::make_shared<LayerNorm>(embed_dim)),
          mhsa(std::make_shared<MultiHeadSelfAttention>(embed_dim, num_heads, dropout_rate)),
          ln2(std::make_shared<LayerNorm>(embed_dim)),
          ff(std::make_shared<FeedForward>(embed_dim, ff_dim, act_type, dropout_rate)) {}

    Tensor forward(const Tensor& input, bool training = false) override {
        Tensor x = ln1->forward(input, training);
        mhsa_cache = mhsa->forward(x, training);

        //std::cout << "TransformerEncoder: input shape {";
        //for (int s : input.shape) std::cout << s << ",";
        //std::cout << "}, mhsa_cache shape {";
        //for (int s : mhsa_cache.shape) std::cout << s << ",";
        //std::cout << "}\n";
        
        x = input + mhsa_cache;
        ff_cache = ff->forward(ln2->forward(x, training), training);
        
        //std::cout << "TransformerEncoder: x shape {";
        //for (int s : x.shape) std::cout << s << ",";
        //std::cout << "}, ff_cache shape {";
        //for (int s : ff_cache.shape) std::cout << s << ",";
        //std::cout << "}\n";

        return x + ff_cache;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_ff = ln2->backward(grad_output);
        grad_ff = ff->backward(grad_ff);
        Tensor grad_x = grad_output + grad_ff;
        grad_x = ln1->backward(mhsa->backward(grad_x));
        return grad_x + grad_output;
    }

    Tensor backward(const Tensor& grad_output) override {
        // Paso 1: Gradiente con respecto a la suma final (x + FF)
        Tensor grad_ff_out = grad_output;

        // Paso 2: Retropropagación a través del bloque FFN
        Tensor grad_ff_in = ff->backward(grad_ff_out);           // dL/d(ff_input)
        Tensor grad_ln2 = ln2->backward(grad_ff_in);             // dL/d(post_mhsa)

        // Paso 3: Sumar con el skip connection de la salida de MHSA
        Tensor grad_post_mhsa = grad_ln2 + grad_output;

        // Paso 4: Retropropagación a través del MHSA
        Tensor grad_mhsa = mhsa->backward(grad_post_mhsa);       // dL/d(LN1_out)
        Tensor grad_ln1 = ln1->backward(grad_mhsa);              // dL/d(input)

        // Nota: no es necesario sumar otra vez grad_output aquí, ya se usó
        return grad_ln1;
    }
        
    void update_weights(Optimizer* optimizer) override {
        ln1->update_weights(optimizer);
        mhsa->update_weights(optimizer);
        ln2->update_weights(optimizer);
        ff->update_weights(optimizer);
    }

    size_t num_params() const override {
        return ln1->num_params() + mhsa->num_params() + ln2->num_params() + ff->num_params();
    }
};
*/