
class MultiHeadSelfAttention : public Layer {
private:
    int embed_dim;
    int num_heads;
    Tensor W_q, W_k, W_v, W_o;
    Tensor bias_q, bias_k, bias_v, bias_o;
    Tensor q_cache, k_cache, v_cache, attn_cache, input_cache, out_2d_final_cache;
    float dropout_rate;
    Tensor mask;
    std::mt19937 rng;
    Tensor W_q_grad, W_k_grad, W_v_grad, W_o_grad;
    Tensor bias_q_grad, bias_k_grad, bias_v_grad, bias_o_grad;
    
    void initialize_weights() {
        int head_dim = embed_dim / num_heads;
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim));
        std::vector<float> w_data(embed_dim * embed_dim);
        std::vector<float> b_data(embed_dim);
        for (float& w : w_data) w = dist(rng);
        for (float& b : b_data) b = 0.0f;
        W_q = Tensor(w_data, {embed_dim, embed_dim});
        W_k = Tensor(w_data, {embed_dim, embed_dim});
        W_v = Tensor(w_data, {embed_dim, embed_dim});
        W_o = Tensor(w_data, {embed_dim, embed_dim});
        bias_q = Tensor(b_data, {embed_dim});
        bias_k = Tensor(b_data, {embed_dim});
        bias_v = Tensor(b_data, {embed_dim});
        bias_o = Tensor(b_data, {embed_dim});
    }
    /*
    void initialize_weights() {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim));

        auto init_weight = [&](Tensor& W) {
            std::vector<float> data(embed_dim * embed_dim);
            for (float& w : data) w = dist(rng);
            W = Tensor(data, {embed_dim, embed_dim});
        };

        auto init_bias = [&](Tensor& b) {
            std::vector<float> data(embed_dim, 0.0f);
            b = Tensor(data, {embed_dim});
        };

        init_weight(W_q);
        init_weight(W_k);
        init_weight(W_v);
        init_weight(W_o);

        init_bias(bias_q);
        init_bias(bias_k);
        init_bias(bias_v);
        init_bias(bias_o);
    }
    */

public:
    MultiHeadSelfAttention(int embed_dim, int num_heads, float dropout_rate = 0.0f)
        : embed_dim(embed_dim), num_heads(num_heads), dropout_rate(dropout_rate), rng(std::random_device{}()) {
        if (embed_dim % num_heads != 0) {
            throw std::runtime_error("embed_dim must be divisible by num_heads");
        }
        initialize_weights();
    }

    Tensor forward(const Tensor& input, bool training = false) override {
        int batch_size = input.shape[0];
        int seq_len = input.shape[1];
        int head_dim = embed_dim / num_heads;
        input_cache = input;

        // Reshape input to 2D for matrix multiplication
        Tensor input_2d(input.data, {batch_size * seq_len, embed_dim});
        Tensor Q = input_2d.matmul(W_q) + bias_q;
        Tensor K = input_2d.matmul(W_k) + bias_k;
        Tensor V = input_2d.matmul(W_v) + bias_v;

        // Reshape Q, K, V to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> Q_new(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> K_new(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> V_new(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        Q_new[idx] = Q.data[src_idx];
                        K_new[idx] = K.data[src_idx];
                        V_new[idx] = V.data[src_idx];
                    }
                }
            }
        }
        Q = Tensor(Q_new, {batch_size, seq_len, num_heads, head_dim});
        K = Tensor(K_new, {batch_size, seq_len, num_heads, head_dim});
        V = Tensor(V_new, {batch_size, seq_len, num_heads, head_dim});
        q_cache = Q;
        k_cache = K;
        v_cache = V;

        // Reshape Q, K to [batch_size * num_heads, seq_len, head_dim]
        std::vector<float> Q_attn(batch_size * num_heads * seq_len * head_dim);
        std::vector<float> K_attn(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        Q_attn[idx] = Q.data[src_idx];
                        K_attn[idx] = K.data[src_idx];
                    }
                }
            }
        }
        Tensor Q_2d(Q_attn, {batch_size * num_heads, seq_len, head_dim});
        Tensor K_2d(K_attn, {batch_size * num_heads, seq_len, head_dim});

        // Transpose K to [batch_size * num_heads, head_dim, seq_len]
        std::vector<float> K_trans_data(batch_size * num_heads * head_dim * seq_len);
        for (int n = 0; n < batch_size * num_heads; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int d = 0; d < head_dim; ++d) {
                    K_trans_data[n * head_dim * seq_len + d * seq_len + p] = K_2d.data[n * seq_len * head_dim + p * head_dim + d];
                }
            }
        }
        Tensor K_trans(K_trans_data, {batch_size * num_heads, head_dim, seq_len});

        // Compute scores
        Tensor scores_2d = Q_2d.matmul(K_trans) / std::sqrt((float)head_dim);

        //std::cout << "MHSA scores_2d shape: {";
        //for (int s : scores_2d.shape) std::cout << s << ",";
        //std::cout << "}\n";

        // Reshape scores to [batch_size, num_heads, seq_len, seq_len]
        std::vector<float> scores_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        int src_idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                        scores_data[idx] = scores_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor scores(scores_data, {batch_size, num_heads, seq_len, seq_len});

        // Softmax and dropout
        scores = scores.softmax();
        attn_cache = scores;
        Tensor scores_2d_attn;

        if (dropout_rate > 0.0f && training) {
            // Reshape scores to [batch_size * num_heads, seq_len, seq_len] for dropout
            std::vector<float> scores_2d_data(batch_size * num_heads * seq_len * seq_len);
            for (int n = 0; n < batch_size; ++n) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int i = 0; i < seq_len; ++i) {
                        for (int j = 0; j < seq_len; ++j) {
                            int idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                            int src_idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            scores_2d_data[idx] = scores.data[src_idx];
                        }
                    }
                }
            }
            Tensor scores_2d_for_dropout(scores_2d_data, {batch_size * num_heads, seq_len, seq_len});
            mask = scores_2d_for_dropout.dropout_mask(dropout_rate, rng);

            //std::cout << "MHSA mask shape: {";
            //for (int s : mask.shape) std::cout << s << ",";
            //std::cout << "}\n";


            scores_2d_attn = scores_2d_for_dropout * mask;
        } else {
            scores_2d_attn = Tensor(scores_2d.data, {batch_size * num_heads, seq_len, seq_len});
        }

        // Compute attention output
        std::vector<float> V_attn(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        V_attn[idx] = V.data[src_idx];
                    }
                }
            }
        }
        Tensor V_2d(V_attn, {batch_size * num_heads, seq_len, head_dim});

        Tensor out_2d = scores_2d_attn.matmul(V_2d);

        // Reshape output to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> out_data(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        out_data[idx] = out_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor out(out_data, {batch_size, seq_len, num_heads, head_dim});

        // Reshape to [batch_size * seq_len, embed_dim]
        std::vector<float> final_out_data(batch_size * seq_len * embed_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        final_out_data[idx] = out.data[src_idx];
                    }
                }
            }
        }
        Tensor out_2d_final(final_out_data, {batch_size * seq_len, embed_dim});
        out_2d_final_cache = out_2d_final; // Cache for backward
        Tensor result = out_2d_final.matmul(W_o) + bias_o;

        // Reshape to [batch_size, seq_len, embed_dim]
        std::vector<float> output_data(batch_size * seq_len * embed_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int d = 0; d < embed_dim; ++d) {
                    int idx = n * seq_len * embed_dim + p * embed_dim + d;
                    int src_idx = (n * seq_len + p) * embed_dim + d;
                    output_data[idx] = result.data[src_idx];
                }
            }
        }
        Tensor output(output_data, {batch_size, seq_len, embed_dim});

        //std::cout << "MHSA output shape: {";
        //for (int s : output.shape) std::cout << s << ",";
        //std::cout << "}\n";
        
        return output;
    }
    Tensor backward(const Tensor& grad_output) override {
        int batch_size = grad_output.shape[0];
        int seq_len = grad_output.shape[1];
        int head_dim = embed_dim / num_heads;

        // Reshape grad_output to [batch_size * seq_len, embed_dim]
        Tensor grad_output_2d(grad_output.data, {batch_size * seq_len, embed_dim});

        // Backward through W_o and bias_o
        W_o_grad = out_2d_final_cache.transpose().matmul(grad_output_2d);
        std::vector<float> bias_o_grad_data(embed_dim, 0.0f);
        for (int i = 0; i < batch_size * seq_len; ++i) {
            for (int d = 0; d < embed_dim; ++d) {
                bias_o_grad_data[d] += grad_output_2d.data[i * embed_dim + d];
            }
        }
        bias_o_grad = Tensor(bias_o_grad_data, {embed_dim});
        Tensor grad_out_2d = grad_output_2d.matmul(W_o.transpose());

        // Reshape grad_out_2d to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> grad_out_data(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        grad_out_data[idx] = grad_out_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_out(grad_out_data, {batch_size, seq_len, num_heads, head_dim});

        // Reshape grad_out to [batch_size * num_heads, seq_len, head_dim]
        std::vector<float> grad_out_2d_data(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        grad_out_2d_data[idx] = grad_out.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_out_2d_reshaped(grad_out_2d_data, {batch_size * num_heads, seq_len, head_dim});

        // Reshape attn_cache to [batch_size * num_heads, seq_len, seq_len]
        std::vector<float> attn_2d_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                        int src_idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        attn_2d_data[idx] = attn_cache.data[src_idx];
                    }
                }
            }
        }
        Tensor attn_2d(attn_2d_data, {batch_size * num_heads, seq_len, seq_len});

        // Backward through attention: grad_scores_2d = grad_out_2d_reshaped * attn_2d^T
        Tensor grad_scores_2d = attn_2d.transpose().matmul(grad_out_2d_reshaped);

        //std::cout << "Backward grad_scores_2d shape: {";
        //for (int s : grad_scores_2d.shape) std::cout << s << ",";
        //std::cout << "} mask shape: {";
        //for (int s : mask.shape) std::cout << s << ",";
        //std::cout << "}\n";

        // Apply dropout mask
        if (dropout_rate > 0.0f) {
            grad_scores_2d = grad_scores_2d * mask;
        }

        // Softmax backward
        std::vector<float> grad_attn_2d_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size * num_heads; ++n) {
            for (int i = 0; i < seq_len; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    int idx = n * seq_len * seq_len + i * seq_len + j;
                    sum += attn_2d.data[idx] * grad_scores_2d.data[idx];
                }
                for (int j = 0; j < seq_len; ++j) {
                    int idx = n * seq_len * seq_len + i * seq_len + j;
                    grad_attn_2d_data[idx] = grad_scores_2d.data[idx] - sum * attn_2d.data[idx];
                }
            }
        }
        Tensor grad_attn_2d(grad_attn_2d_data, {batch_size * num_heads, seq_len, seq_len});

        // Reshape grad_attn_2d to [batch_size, num_heads, seq_len, seq_len]
        std::vector<float> grad_attn_data(batch_size * num_heads * seq_len * seq_len);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        int idx = n * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        int src_idx = (n * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                        grad_attn_data[idx] = grad_attn_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_attn(grad_attn_data, {batch_size, num_heads, seq_len, seq_len});

        // Reshape q_cache, k_cache, v_cache to [batch_size * num_heads, seq_len, head_dim]
        std::vector<float> q_2d_data(batch_size * num_heads * seq_len * head_dim);
        std::vector<float> k_2d_data(batch_size * num_heads * seq_len * head_dim);
        std::vector<float> v_2d_data(batch_size * num_heads * seq_len * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        q_2d_data[idx] = q_cache.data[src_idx];
                        k_2d_data[idx] = k_cache.data[src_idx];
                        v_2d_data[idx] = v_cache.data[src_idx];
                    }
                }
            }
        }
        Tensor Q_2d(q_2d_data, {batch_size * num_heads, seq_len, head_dim});
        Tensor K_2d(k_2d_data, {batch_size * num_heads, seq_len, head_dim});
        Tensor V_2d(v_2d_data, {batch_size * num_heads, seq_len, head_dim});

        // Backward through Q, K, V
        Tensor grad_Q_2d = grad_attn_2d.matmul(K_2d) / std::sqrt((float)head_dim);
        Tensor grad_K_2d = grad_attn_2d.transpose().matmul(Q_2d) / std::sqrt((float)head_dim);
        Tensor grad_V_2d = attn_2d.matmul(grad_out_2d_reshaped);

        // Reshape gradients to [batch_size, seq_len, num_heads, head_dim]
        std::vector<float> grad_Q_data(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> grad_K_data(batch_size * seq_len * num_heads * head_dim);
        std::vector<float> grad_V_data(batch_size * seq_len * num_heads * head_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < num_heads; ++h) {
                for (int p = 0; p < seq_len; ++p) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        int src_idx = (n * num_heads + h) * seq_len * head_dim + p * head_dim + d;
                        grad_Q_data[idx] = grad_Q_2d.data[src_idx];
                        grad_K_data[idx] = grad_K_2d.data[src_idx];
                        grad_V_data[idx] = grad_V_2d.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_Q(grad_Q_data, {batch_size, seq_len, num_heads, head_dim});
        Tensor grad_K(grad_K_data, {batch_size, seq_len, num_heads, head_dim});
        Tensor grad_V(grad_V_data, {batch_size, seq_len, num_heads, head_dim});

        // Reshape to [batch_size * seq_len, embed_dim]
        std::vector<float> grad_Q_2d_final(batch_size * seq_len * embed_dim);
        std::vector<float> grad_K_2d_final(batch_size * seq_len * embed_dim);
        std::vector<float> grad_V_2d_final(batch_size * seq_len * embed_dim);
        for (int n = 0; n < batch_size; ++n) {
            for (int p = 0; p < seq_len; ++p) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        int idx = n * seq_len * embed_dim + p * embed_dim + h * head_dim + d;
                        int src_idx = n * seq_len * num_heads * head_dim + p * num_heads * head_dim + h * head_dim + d;
                        grad_Q_2d_final[idx] = grad_Q.data[src_idx];
                        grad_K_2d_final[idx] = grad_K.data[src_idx];
                        grad_V_2d_final[idx] = grad_V.data[src_idx];
                    }
                }
            }
        }
        Tensor grad_Q_final(grad_Q_2d_final, {batch_size * seq_len, embed_dim});
        Tensor grad_K_final(grad_K_2d_final, {batch_size * seq_len, embed_dim});
        Tensor grad_V_final(grad_V_2d_final, {batch_size * seq_len, embed_dim});

        // Compute gradients for weights
        Tensor input_2d(input_cache.data, {batch_size * seq_len, embed_dim});
        W_q_grad = input_2d.transpose().matmul(grad_Q_final);
        W_k_grad = input_2d.transpose().matmul(grad_K_final);
        W_v_grad = input_2d.transpose().matmul(grad_V_final);
        std::vector<float> bias_q_grad_data(embed_dim, 0.0f);
        std::vector<float> bias_k_grad_data(embed_dim, 0.0f);
        std::vector<float> bias_v_grad_data(embed_dim, 0.0f);
        for (int i = 0; i < batch_size * seq_len; ++i) {
            for (int d = 0; d < embed_dim; ++d) {
                bias_q_grad_data[d] += grad_Q_final.data[i * embed_dim + d];
                bias_k_grad_data[d] += grad_K_final.data[i * embed_dim + d];
                bias_v_grad_data[d] += grad_V_final.data[i * embed_dim + d];
            }
        }
        bias_q_grad = Tensor(bias_q_grad_data, {embed_dim});
        bias_k_grad = Tensor(bias_k_grad_data, {embed_dim});
        bias_v_grad = Tensor(bias_v_grad_data, {embed_dim});

        // Compute input gradient
        Tensor grad_input_2d = grad_Q_final.matmul(W_q.transpose()) + grad_K_final.matmul(W_k.transpose()) + grad_V_final.matmul(W_v.transpose());

        // Reshape to [batch_size, seq_len, embed_dim]
        Tensor grad_input(grad_input_2d.data, {batch_size, seq_len, embed_dim});
        return grad_input;
    }

    void update_weights(Optimizer* optimizer) override {
        optimizer->update(W_q, W_q_grad);
        optimizer->update(W_k, W_k_grad);
        optimizer->update(W_v, W_v_grad);
        optimizer->update(W_o, W_o_grad);
        optimizer->update(bias_q, bias_q_grad);
        optimizer->update(bias_k, bias_k_grad);
        optimizer->update(bias_v, bias_v_grad);
        optimizer->update(bias_o, bias_o_grad);
    }

    size_t num_params() const override {
        return W_q.total_elements() + W_k.total_elements() + W_v.total_elements() + W_o.total_elements() +
               bias_q.total_elements() + bias_k.total_elements() + bias_v.total_elements() + bias_o.total_elements();
    }
};
