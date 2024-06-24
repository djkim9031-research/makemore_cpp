#include "nn_models.h"

namespace{
    std::vector<std::string> simple_name_generator(std::unordered_map<char, int>& stoi,
                                                   std::unordered_map<int, char>& itos, 
                                                   const torch::Tensor& trained_weights,
                                                   const int num_names){
        std::vector<std::string> generated_names;
        auto gen = at::detail::createCPUGenerator(42);
        for(int n=0; n<num_names;++n){
            std::string name = "";
            char curr_char = '.';

            int idx = 0;
            while(true){
                auto x_infer_tensor = torch::tensor({stoi[curr_char]}, torch::kInt64);
                auto xenc = torch::nn::functional::one_hot(x_infer_tensor, 27).to(torch::kFloat32);
                auto logits = torch::matmul(xenc, trained_weights);
                auto counts = logits.exp(); // 1 x 27
                auto probs = counts / counts.sum(1, true);

                idx = torch::multinomial(probs, /*num_samples=*/1, /*replacement=*/true, gen).item<int>();
                curr_char = itos[idx];
                if(idx == 0){
                    break;
                }
                name += itos[idx];
            }
            generated_names.push_back(name);
        }
        
        return generated_names;
    }

    void build_dataset(const int context_win_size,
                      const std::unordered_map<char, int>& stoi,
                      const std::vector<std::string>& words,
                      torch::Tensor& Xs,
                      torch::Tensor& Ys){
        std::vector<int64_t> xs; // num_existing_char x context_win_size
        std::vector<int64_t> ys; // num_exisiting_char
        for(const std::string& word : words){
            std::vector<int> context;
            for(size_t i=0; i<context_win_size; ++i){
                context.push_back(0);
            }

            std::string curr_name = word + ".";
            for(char ch : curr_name){
                int idx = stoi.at(ch);
                xs.insert(xs.end(), context.begin(), context.end());
                ys.push_back(idx);

                for(int i=0; i<context_win_size - 1; ++i){
                    context[i] = context[i+1];
                }
                context[context_win_size - 1] = idx;

            }
        }

        Xs = torch::from_blob(xs.data(), {static_cast<int>(ys.size()), context_win_size}, torch::kInt64).clone();
        Ys = torch::tensor(ys, torch::kInt64);
    }

    float evaluation(const torch::Tensor& validation_data,
                     const torch::Tensor& validation_target,
                     const int context_win_size,
                     const int vocab_size,
                     const std::vector<std::unique_ptr<Layer>>& layers){

        auto out = torch::nn::functional::one_hot(validation_data, vocab_size).to(torch::kFloat32);
        for(size_t i=0; i<layers.size(); ++i){
            layers[i]->eval();
            out = layers[i]->forward(out);
        }

        auto loss = torch::nn::functional::cross_entropy(out, validation_target);
        return loss.item<float>();
    }

    std::vector<std::string> nn_name_inference(const std::unordered_map<char, int>& stoi,
                                               const std::unordered_map<int, char>& itos,
                                               const int num_names,
                                               const int context_win_size,
                                               const int vocab_size,
                                               const std::vector<std::unique_ptr<Layer>>& layers,
                                               const torch::Generator& gen){
        std::vector<std::string> generated_names;
        for(int n=0; n<num_names; ++n){
            std::string name = "";
            char curr_char = '.';

            int idx = 0;
            auto xs_tensor = torch::zeros({1, context_win_size}, torch::kInt64);
            while(true){
                // Shift elements to the right
                auto temp_tensor = xs_tensor.slice(1, 1, context_win_size).clone();
                xs_tensor.slice(1, 0, context_win_size-1) = temp_tensor;
                // Set the last element to the new value.
                xs_tensor[0][context_win_size-1] = stoi.at(curr_char);

                auto out = torch::nn::functional::one_hot(xs_tensor, vocab_size).to(torch::kFloat32);
                for(size_t i=0; i<layers.size(); ++i){
                    layers[i]->eval();
                    out = layers[i]->forward(out);
                }
                auto counts = out.exp();  // num_names, vocab_size
                auto probs = counts / counts.sum(1, true);

                idx = torch::multinomial(probs, /*num_samples=*/1, /*replacement=*/true, gen).item<int>();
                curr_char = itos.at(idx);
                curr_char = itos.at(idx);
                if(idx == 0){
                    break;
                }
                name += itos.at(idx);

            }
            generated_names.push_back(name);
        }
        return generated_names;
    }

    // This estimates the bn_mean based on the entire training dataset.
    // It cumbersome and could potentially be computationally very expensive to call this
    // at each evaluation time during the training process.
    // Currently implementation does not use it.
    void calibrate_batch_norm(const torch::Tensor& xs_tensor,
                              const int context_win_size,
                              const int vocab_size,
                              const std::vector<torch::Tensor*>& trained_params,
                              std::vector<torch::Tensor>& bn_params){
        torch::NoGradGuard no_grad; // RAII-style no_grad lock.

        bn_params.clear();

        auto xenc = torch::nn::functional::one_hot(xs_tensor, vocab_size).to(torch::kFloat32);
        auto emb = torch::matmul(xenc, *trained_params[0]); 
        auto out = emb.view({emb.sizes()[0], emb.sizes()[1] * emb.sizes()[2]});
        for(size_t i=1; i<trained_params.size()-2; i+=4){
            auto pre_act = torch::matmul(out, *trained_params[i]) + (*trained_params[i+1]);

            auto bn_mean = pre_act.mean(0, /*keepdim=*/true);
            auto bn_std = pre_act.std(0, /*unbiased=*/true, /*keepdim=*/true);

            bn_params.push_back(bn_mean);
            bn_params.push_back(bn_std);

            pre_act = ((*trained_params[i+2])*(pre_act - bn_mean) / (bn_std) ) + (*trained_params[i+3]);

            out = torch::tanh(pre_act);
        }

    }
} // end of namespace

void simple_neuron_model(const std::string& data_path, int num_names){

    auto N = torch::ones({27, 27}, torch::kInt32);

    std::vector<std::string> words;
    std::unordered_map<int, char> itos;
    std::unordered_map<char, int> stoi;
    tokenizer(data_path, N, words, itos, stoi);

    std::vector<int> xs, ys;
    for(const std::string& word : words){
        std::string chs = "."+word+".";
        for(size_t i=0; i<chs.size()-1; ++i){
            int idx1 = stoi[chs[i]];
            int idx2 = stoi[chs[i+1]];
            xs.push_back(idx1);
            ys.push_back(idx2);
        }
    }

    // Convert to torch tensors.
    auto xs_tensor = torch::tensor(xs, torch::kInt64);
    auto ys_tensor = torch::tensor(ys, torch::kInt64);

    // Initialize weights.
    auto gen = at::detail::createCPUGenerator(42);
    auto w = torch::randn({27, 27}, gen).set_requires_grad(true);

    // One-hot encoding of xs_tensor
    auto xenc = torch::nn::functional::one_hot(xs_tensor, 27).to(torch::kFloat32);
    int nEvents = xs.size();

    // Prior to training, generated name:
    std::cout<<"Generated name(s) prior to training: ";
    std::vector<std::string> gen_names = simple_name_generator(stoi, itos, w, num_names);
    for(int n=0; n<num_names; ++n){
        std::cout<<gen_names[n]<<"   ";
    }
    std::cout<<"\n";
    std::cout<<"________________________________________________________"<<std::endl;

    // Training loop
    int iter = 0;
    while(true){
        // Forward pass
        auto logits = torch::matmul(xenc, w);
        auto counts = logits.exp();
        auto probs = counts / counts.sum(1, true); //(nEvents, 27)

        // Loss calculation (negative log likelihood)
        auto idx = torch::arange(nEvents);
        auto indexed_probs = probs.index({idx, ys_tensor});
        auto loss = -indexed_probs.log().mean() + 0.0001*w.pow(2).mean(); //nll loss with L2 regularization for smoothing

        if(iter%10==0){
            std::cout<<"[Iteration "<<iter<<", loss = "<<loss.item<float>()<<"]"<<std::endl;
            std::cout<<"[Inference - simple neuron] Generated name(s): ";
            gen_names.clear();
            gen_names = simple_name_generator(stoi, itos, w, num_names);
            for(int n=0; n<num_names; ++n){
                std::cout<<gen_names[n]<<"   ";
            }
            std::cout<<"\n";
            std::cout<<"________________________________________________________"<<std::endl;
        }
        iter += 1;
        if(loss.item<float>() < 2.47){ //Roughly the nll loss of bigram
            break;
        }

        // Backward pass
        if (w.grad().defined()) {
            w.grad().zero_();
        }
        loss.backward();

        // Update parameters
        w.data() -= 50 * w.grad();
    }
    return;
}


void mlp_model(const std::string& data_path, const int context_win_size, int num_names, int batch_size){
    
    auto gen = at::detail::createCPUGenerator(42);
    int embedding_space_dim = 10;
    int vocab_size = 27;
    int n_hidden = 200;

    auto N = torch::ones({vocab_size, vocab_size}, torch::kInt32);

    std::vector<std::string> words;
    std::unordered_map<int, char> itos;
    std::unordered_map<char, int> stoi;
    tokenizer(data_path, N, words, itos, stoi);

    //________________________________________________________________________________
    // Data preparation and create training data
    //________________________________________________________________________________
    // Seed the random number generator
    std::mt19937 rng(42); // 42 is the seed value

    // Shuffle the words
    std::shuffle(words.begin(), words.end(), rng);

    // Calculate n1 and n2
    int n1 = static_cast<int>(0.8 * words.size());
    int n2 = static_cast<int>(0.9 * words.size());

    std::vector<std::string> training_data(words.begin(), words.begin() + n1);
    std::vector<std::string> validation_data(words.begin() + n1, words.begin() + n2);
    std::vector<std::string> test_data(words.begin() + n2, words.end());

    torch::Tensor xs_tensor, ys_tensor, xval_tensor, yval_tensor, xtest_tensor, ytest_tensor;

    build_dataset(context_win_size, stoi, training_data, xs_tensor, ys_tensor);
    build_dataset(context_win_size, stoi, validation_data, xval_tensor, yval_tensor);
    build_dataset(context_win_size, stoi, test_data, xtest_tensor, ytest_tensor);


    //________________________________________________________________________________
    // Layers definition
    //________________________________________________________________________________
    auto embedding = std::make_unique<Linear>(vocab_size, embedding_space_dim, "embedding", gen, false);
    auto ln1 = std::make_unique<Linear>(context_win_size*embedding_space_dim, n_hidden, "dense1", gen, false);
    auto bn1 = std::make_unique<BatchNorm1D>(n_hidden, "batch_norm1", 0.1);
    auto tanh1 = std::make_unique<TanhActivation>("tanh1");

    auto ln2 = std::make_unique<Linear>(n_hidden, 2*n_hidden, "dense2", gen, false);
    auto bn2 = std::make_unique<BatchNorm1D>(2*n_hidden, "batch_norm2", 0.1);
    auto tanh2 = std::make_unique<TanhActivation>("tanh2");

    auto ln3 = std::make_unique<Linear>(2*n_hidden, n_hidden, "dense3", gen, false);
    auto bn3 = std::make_unique<BatchNorm1D>(n_hidden, "batch_norm3", 0.1);
    auto tanh3 = std::make_unique<TanhActivation>("tanh3");

    auto ln4 = std::make_unique<Linear>(n_hidden, vocab_size, "dense4", gen, true);

    std::vector<std::unique_ptr<Layer>> mlp_layers;
    mlp_layers.push_back(std::move(embedding));
    mlp_layers.push_back(std::move(ln1));
    mlp_layers.push_back(std::move(bn1));
    mlp_layers.push_back(std::move(tanh1));

    mlp_layers.push_back(std::move(ln2));
    mlp_layers.push_back(std::move(bn2));
    mlp_layers.push_back(std::move(tanh2));

    mlp_layers.push_back(std::move(ln3));
    mlp_layers.push_back(std::move(bn3));
    mlp_layers.push_back(std::move(tanh3));

    mlp_layers.push_back(std::move(ln4));

    std::vector<torch::Tensor*> params = {};
    for(size_t i=0; i<mlp_layers.size(); ++i){
        std::vector<torch::Tensor*> curr_layer_params = mlp_layers[i]->parameters();
        params.insert(params.end(), curr_layer_params.begin(), curr_layer_params.end());
    }

    //________________________________________________________________________________
    // Training
    //________________________________________________________________________________

    // Prior to training, generated name:
    std::cout<<"Generated name(s) prior to training: ";
    std::vector<std::string> gen_names = nn_name_inference(stoi, itos, num_names, context_win_size, vocab_size, mlp_layers, gen);
    for(int n=0; n<num_names; ++n){
        std::cout<<gen_names[n]<<"   ";
    }
    std::cout<<"\n";
    std::cout<<"________________________________________________________"<<std::endl;

    int iter = 0;
    while(true){

        // Mini batch
        auto idx = torch::randint(0, xs_tensor.size(0), {batch_size}, torch::kInt64); 
        auto xs_tensor_batch = xs_tensor.index_select(0, idx);
        auto ys_tensor_batch = ys_tensor.index_select(0, idx);
        auto out = torch::nn::functional::one_hot(xs_tensor_batch, vocab_size).to(torch::kFloat32);

        // Forward pass
        for(size_t i=0; i<mlp_layers.size(); ++i){
            mlp_layers[i]->train();
            out = mlp_layers[i]->forward(out);
        }

        auto loss = torch::nn::functional::cross_entropy(out, ys_tensor_batch);

        if(iter%10000==0){
            //calibrate_batch_norm(xs_tensor, context_win_size, vocab_size, params, bn_params);
            std::cout<<"[Iteration "<<iter<<", Training loss = "<<loss.item<float>()<<"]"<<std::endl;
            std::cout<<"[Evaluation loss = "<<evaluation(xval_tensor, yval_tensor, context_win_size, vocab_size, mlp_layers)<<"]"<<std::endl;
            std::cout<<"[Inference - MLP model] Generated name(s): ";
            gen_names.clear();
            gen_names = nn_name_inference(stoi, itos, num_names, context_win_size, vocab_size, mlp_layers, gen);
            for(int n=0; n<num_names; ++n){
                std::cout<<gen_names[n]<<"   ";
            }
            std::cout<<"\n";
            std::cout<<"________________________________________________________"<<std::endl;
        }
        iter += 1;
        if(iter>200000){
            break;
        }

        // Backward pass
        for(size_t i=0; i<params.size(); ++i){
            if(params[i]->grad().defined()){
                params[i]->grad().zero_();
            }
        }
        loss.backward();

        // Update parameters
        //int lr_idx = (iter_10s < num_spacings) ? num_spacings - 1- iter_10s : 0;
        //float lr = lrs[lr_idx].item<float>();
        float lr = (iter > 100000) ? 0.01 : 0.1;
        for(size_t i=0; i<params.size(); ++i){
            params[i]->data() -= lr*params[i]->grad();
        }
    }

    // To visualize what embedding space has learned, use the following visualizer.
    // NOTE: Since any arbitrary n dimension cannot be plotted, embedding_space_dim == 2
    // must be satisfied when calling this function.
    //viz_embedding_space(mlp_layers[0]->weights(), itos);

    // To visualize the tanh activation output distribution, call the following function.
    // Generally, with Kaimeng initialization and/or with Batch normalization, the saturation
    // will be 0.
    // Without proper initialization or batch normalization, saturation will be more pronounced.
    viz_tanh_activation_dist(mlp_layers); 
}


void custom_backprop_test(const std::string& data_path, const int context_win_size, int num_names, int batch_size){

    auto gen = at::detail::createCPUGenerator(42);
    int embedding_space_dim = 10;
    int vocab_size = 27;
    int n_hidden = 200;

    auto N = torch::ones({vocab_size, vocab_size}, torch::kInt32);

    std::vector<std::string> words;
    std::unordered_map<int, char> itos;
    std::unordered_map<char, int> stoi;
    tokenizer(data_path, N, words, itos, stoi);

    //________________________________________________________________________________
    // Data preparation and create training data
    //________________________________________________________________________________
    // Seed the random number generator
    std::mt19937 rng(42); // 42 is the seed value

    // Shuffle the words
    std::shuffle(words.begin(), words.end(), rng);

    // Calculate n1 and n2
    int n1 = static_cast<int>(0.8 * words.size());
    int n2 = static_cast<int>(0.9 * words.size());

    std::vector<std::string> training_data(words.begin(), words.begin() + n1);
    std::vector<std::string> validation_data(words.begin() + n1, words.begin() + n2);
    std::vector<std::string> test_data(words.begin() + n2, words.end());

    torch::Tensor xs_tensor, ys_tensor, xval_tensor, yval_tensor, xtest_tensor, ytest_tensor;

    build_dataset(context_win_size, stoi, training_data, xs_tensor, ys_tensor);
    build_dataset(context_win_size, stoi, validation_data, xval_tensor, yval_tensor);
    build_dataset(context_win_size, stoi, test_data, xtest_tensor, ytest_tensor);

    //________________________________________________________________________________
    // Definition of trainable parameters
    //________________________________________________________________________________

    auto C = torch::randn({vocab_size, embedding_space_dim}, gen);
    auto W1 = torch::randn({embedding_space_dim*context_win_size, n_hidden}, gen)*5/(3*std::sqrt(embedding_space_dim*context_win_size));
    auto b1 = torch::randn(n_hidden, gen)*0.1;
    auto W2 = torch::randn({n_hidden, vocab_size}, gen)*0.1;
    auto b2 = torch::randn(vocab_size, gen)*0.1;

    auto bn_gain = torch::randn({1, n_hidden})*0.1 + 1.0;
    auto bn_bias = torch::rand({1, n_hidden})*0.1;

    std::vector<torch::Tensor*> params = {&C, &W1, &b1, &bn_gain, &bn_bias, &W2, &b2};
    for(torch::Tensor* param : params){
        param->set_requires_grad(true);
    }


    //________________________________________________________________________________
    // Batch selection
    //________________________________________________________________________________

    auto idx = torch::randint(0, xs_tensor.size(0), {batch_size}, torch::kInt64);
    auto xs_tensor_batch = xs_tensor.index_select(0, idx);
    auto ys_tensor_batch = ys_tensor.index_select(0, idx);

    //________________________________________________________________________________
    // Layers
    //________________________________________________________________________________

    auto xenc = torch::nn::functional::one_hot(xs_tensor_batch, vocab_size).to(torch::kFloat32);
    auto emb = torch::matmul(xenc, C);
    auto emb_flattened = emb.view({emb.sizes()[0], emb.sizes()[1] * emb.sizes()[2]});

    // Linear layer 1
    auto h_prebn = torch::matmul(emb_flattened, W1) + b1; // batch, n_hidden

    // Batchnorm layer
    auto bn_meani = h_prebn.sum(0, true)/batch_size;    // 1, n_hidden
    auto bn_diff = h_prebn - bn_meani;   // batch_size, n_hidden
    auto bn_diff2 = torch::pow(bn_diff, 2); // batch_size, n_hidden
    auto bn_var = bn_diff2.sum(0, true)/(batch_size -1); // 1, n_hidden
    auto bn_var_inv = torch::pow(bn_var + 1e-5, -0.5); // 1, n_hidden
    auto bn_raw = bn_diff*bn_var_inv; // batch_size, n_hidden
    auto h_preact = bn_gain*bn_raw + bn_bias; //batch_size, n_hidden

    // Non-linearity
    auto h = torch::tanh(h_preact); // batch_size, n_hidden

    // Linear layer 2
    auto logits = torch::matmul(h, W2) + b2; // batch_size, vocab_size

    // Cross entropy loss
    auto logit_maxes = std::get<0>(logits.max(1, true)); // logit.max returns a tuple where 1st elem = max val, 2nd = corr indices
    auto norm_logits = logits - logit_maxes; // Subtract max for numerical stability // bat, vocab
    auto counts = norm_logits.exp(); // batch_size, vocab_size
    auto counts_sum = counts.sum(1, true); // batch_size, 1
    auto counts_sum_inv = torch::pow(counts_sum, -1); // batch_size, 1
    auto probs = counts * counts_sum_inv; // batch_size, vocab_size
    auto log_probs = probs.log(); // batch_size, vocab_size

    auto range_tensor = torch::arange(batch_size, torch::kInt64);
    auto logprobs_selected = log_probs.index({range_tensor, ys_tensor_batch}); // batch_size, 1
    auto loss = -logprobs_selected.mean();

    for(torch::Tensor* param : params){
        if(param->grad().defined()){
            param->grad().zero_();
        }
    }
    std::vector<torch::Tensor*> ts = {&log_probs, &probs, &counts, &counts_sum, &counts_sum_inv, &norm_logits, &logit_maxes,
                                     &logits, &h, &h_preact, &bn_raw, &bn_var_inv, &bn_var, &bn_diff2, &bn_diff, &bn_meani,
                                     &h_prebn, &emb_flattened, &emb};
    for(auto& t : ts){
        t->set_requires_grad(true);
        t->retain_grad();
    }

    loss.backward({},/* retain_graph= */ true);

    //________________________________________________________________________________
    // Backprop calculation
    //________________________________________________________________________________
    auto d_log_probs = torch::zeros_like(log_probs);
    d_log_probs.index_put_({range_tensor, ys_tensor_batch}, torch::tensor({-1.0/batch_size})); // derivative of mean = 1/batch_size

    //log_probs = ln(probs) => dL/d_probs = d_term_so_far * 1/probs
    auto d_probs = (1.0/probs) * d_log_probs; // derivate of ln(x) = 1/x

    auto d_counts_sum_inv = (counts * d_probs).sum(1, true);
    auto d_counts = counts_sum_inv * d_probs;
    // counts_sum_inv = 1/counts_sum => derivative of counts_sum = -1/(counts_sum)^2
    auto d_counts_sum = -torch::pow(counts_sum, -2) * d_counts_sum_inv;
    d_counts += torch::ones_like(counts) * d_counts_sum;

    // (e^x)' = (e^x) = y
    auto d_norm_logits = counts * d_counts;

    auto d_logit_maxes = (-d_norm_logits).sum(1, true);
    auto d_logits = d_norm_logits.clone();
    // Max elements are backpropagated from d_logit_maxes, and rest are directly from d_norm_logits
    d_logits += torch::nn::functional::one_hot(std::get<1>(logits.max(1)), vocab_size).to(torch::kFloat32) * d_logit_maxes;

    auto d_h = torch::matmul(d_logits, W2.transpose(0, 1)); //(batch_size, n_hidden) = (batch_size, vocab_size)*(n_hidden, vocab_size).T
    auto d_W2 = torch::matmul(h.transpose(0, 1), d_logits);
    auto d_b2 = d_logits.sum(0);

    auto d_h_preact = (1.0 - torch::pow(h, 2))*d_h;
    auto d_bn_gain = (bn_raw * d_h_preact).sum(0, true); //(1, n_hidden) = sum_over_dim_0{(batch_size, n_hidden)*(batch_size, n_hidden)}
    auto d_bn_raw = bn_gain * d_h_preact;
    auto d_bn_bias = d_h_preact.sum(0, true);
    auto d_bn_diff = bn_var_inv * d_bn_raw;
    auto d_bn_var_inv = (bn_diff * d_bn_raw).sum(0, true);
    auto d_bn_var = -0.5*(torch::pow(bn_var + 1e-5, -1.5))*d_bn_var_inv; // d(1/x^0.5) => -0.5*(1/x^1.5)
    auto d_bn_diff2 = (1.0/(batch_size - 1))*torch::ones_like(bn_diff2)*d_bn_var;
    d_bn_diff += (2*bn_diff)*d_bn_diff2;
    auto d_bn_meani = -d_bn_diff.sum(0, true);

    auto d_h_prebn = d_bn_diff.clone();
    d_h_prebn += (1.0/batch_size)*(torch::ones_like(h_prebn) * d_bn_meani);

    auto d_emb_flattened = torch::matmul(d_h_prebn, W1.transpose(0, 1));
    auto d_W1 = torch::matmul(emb_flattened.transpose(0, 1), d_h_prebn);
    auto d_b1 = d_h_prebn.sum(0);
    auto d_emb = d_emb_flattened.view({emb.sizes()});
    
    auto d_C = torch::matmul(xenc.transpose(1, 2), d_emb).sum(0); // (vocab_size, embedding_space_dim) = sum_over_dim_0{(batch, conteext, vocab).T  * (batch, context, emb)}
    

    //________________________________________________________________________________
    // Sanity check - validatate the manually implemented backprob results with torch results
    //________________________________________________________________________________
    
    cmp("log_probs", d_log_probs, log_probs);
    cmp("probs", d_probs, probs);
    cmp("counts_sum_inv", d_counts_sum_inv, counts_sum_inv);
    cmp("counts_sum", d_counts_sum, counts_sum);
    cmp("counts", d_counts, counts);
    cmp("norm_logits", d_norm_logits, norm_logits);
    cmp("logit_maxes", d_logit_maxes, logit_maxes);
    cmp("logits", d_logits, logits);
    cmp("h", d_h, h);
    cmp("W2", d_W2, W2);
    cmp("b2", d_b2, b2);
    cmp("h_preact", d_h_preact, h_preact);
    cmp("bn_gain", d_bn_gain, bn_gain);
    cmp("bn_bias", d_bn_bias, bn_bias);
    cmp("bn_raw", d_bn_raw, bn_raw);
    cmp("bn_var_inv", d_bn_var_inv, bn_var_inv);
    cmp("bn_var", d_bn_var, bn_var);
    cmp("bn_diff2", d_bn_diff2, bn_diff2);
    cmp("bn_diff", d_bn_diff, bn_diff);
    cmp("bn_meani", d_bn_meani, bn_meani);
    cmp("h_prebn", d_h_prebn, h_prebn);
    cmp("emb_flattened", d_emb_flattened, emb_flattened);
    cmp("W1", d_W1, W1);
    cmp("b1", d_b1, b1);
    cmp("emb", d_emb, emb);
    cmp("C", d_C, C);


    //________________________________________________________________________________
    // Optimizing loss function with softmax
    //________________________________________________________________________________
    // Cross entropy loss (optimized)
    auto ce_loss = torch::nn::functional::cross_entropy(logits, ys_tensor_batch);
    std::cout<<"Diff between cross entropy loss (torch) and manually calculated: "<<(ce_loss - loss).item<float>()<<std::endl;
    
    // Cross entropy = -1 * (1*ln(softmax_result_for_exp1) + 1*ln(softmax_result_for_exp2) + ...)/(batch_size)
    // dL/d(softmax) = -(1/softmax)/batch_size
    // d(softmax)/d(logits) = softmax * (1 - softmax)
    // dL/d(logits) = -(1-softmax)/batch_size 

    auto d_logits_optimized = torch::zeros_like(logits);
    d_logits_optimized.index_put_({range_tensor, ys_tensor_batch}, torch::tensor({-1.0/batch_size})); 
    d_logits_optimized += torch::nn::functional::softmax(logits, /*dim=*/1)/batch_size;

    d_logits_optimized.set_requires_grad(true);
    d_logits_optimized.retain_grad();

    cmp("logits_optimized", d_logits_optimized, logits);


    //________________________________________________________________________________
    // Optimizing batch norm backprop caculation
    //________________________________________________________________________________

    auto h_preact_optimized = bn_gain*(h_prebn - h_prebn.mean(0, true))/torch::sqrt(h_prebn.var(0, /*unbiased=*/true, /*keepdim=*/true) + 1e-5) + bn_bias;
    std::cout<<"Diff between batchnorm (torch) and manually calculated: "<<(h_preact - h_preact_optimized).abs().max().item<float>()<<std::endl;

    // batch_norm (y) = bn_gain*(norm) + bn_bias, where norm = (x-x.mean)/[x.var(unbiased)]^0.5
    // dy/d(norm) = bn_gain
    // d(norm)/dx = 1/[x.var(unbiased)]^0.5
    // d(norm)/d(x.mean) = -1/[x.var(unbiased)]^0.5
    // d(x.mean)/d(xi) = 1/batch_size
    // d(norm)/d(var) = -0.5*(x-x.mean)*(x.var(unbiased) + eps)^(-3/2)
    // d(var)/d(xi) = 2*(x-x.mean)/(batch_size - 1) // for the unbiased variance

    // ___________________FORMULA__________________________________
    // So, dy/dx = dy/d(norm) * [d(norm)/dx + d(norm)/d(x.mean)*SIGMA[d(x.mean)/d(xi)] + d(norm)/d(var)*SIGMA[d(var)/d(xi)]]
    
    // ___________________TERM 1__________________________________
    // First term, dy/d(norm) * d(norm)/dx = bn_gain * bn_var_inv. 
    // So dL/dx = dL/dy * dy/dx = bn_gain * bn_var_inv * d_h_preact.

    // ___________________TERM 2__________________________________
    // dy/d(norm)*d(norm)/d(x.mean)*SIGMA[d(x.mean)/d(xi)] 
    // = bn_gain *(-bn_var_inv)* SIGMA[ 1/ batch_size] => Terms outside SIGMA is of size (1, n_hidden)
    // So these terms are constant. However, we need to consider loss term backprogaged from later layers.
    // dL/dx = dL/dy * term 2 so far, where dL/dy = d_h_preact is of size (batch_size, n_hidden) 
    // So, SIGMA[dL/dy * dy/d(norm) * d(norm)/d(x.mean)*d(x.mean)/d(xi)]
    // = -bn_gain * bn_var_inv * d_h_preact.sum(dim=0) / batch_size

    // ___________________TERM 3__________________________________
    // dy/d(norm) * d(norm)/d(var)* SIGMA[d(var)/d(xi)]
    // bn_raw = (x - x.mean)/[x.var(unbiased)]^0.5
    // d(norm)/d(var) = -0.5 * bn_raw * bn_var_inv * [x.var(unbiased)]^-0.5
    // d(var)/d(xi) = 2 * bn_raw * [x.var(unbiased)]^0.5 / (batch_size - 1)
    // dL/dx = dL/dy * dy/d(norm) * d(norm)/d(var) * SIGMA[ d(var)/d(xi)]
    // = d_h_preact * bn_gain *(-bn_raw * bn_var_inv) * SIGMA[bn_raw/(batch_size - 1)]
    // Here, dL/dy term should individually contribute to all elements.
    // So, term3 = bn_gain *(-bn_raw * bn_var_inv)/(batch_size - 1) * SIGMA[d_h_preact *bn_raw]
    // = -bn_gain* bn_var_inv * bn_raw * (bn_raw * d_h_preact).sum(dim=0) /(batch_size -1)

    // Adding altogether,
    // dL/dx = bn_gain * bn_var_inv*(d_h_preact - d_h_preact.sum(dim=0) / batch_size - bn_raw * (bn_raw * d_h_preact).sum(dim=0) /(batch_size -1))

    auto d_prebn_optimized = bn_gain*bn_var_inv*(d_h_preact - d_h_preact.sum(0)/batch_size - bn_raw*(d_h_preact*bn_raw).sum(0)/(batch_size-1));

    d_prebn_optimized.set_requires_grad(true);
    d_prebn_optimized.retain_grad();

    cmp("prebn_optimized", d_prebn_optimized, h_prebn);

}