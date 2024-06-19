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

    std::vector<std::string> nn_name_inference(std::unordered_map<char, int>& stoi,
                                               std::unordered_map<int, char>& itos,
                                               const int num_names,
                                               const int context_win_size,
                                               const std::vector<torch::Tensor*>& trained_params){


        std::vector<std::string> generated_names;
        auto gen = at::detail::createCPUGenerator(42);
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
                xs_tensor[0][context_win_size-1] = stoi[curr_char];

                auto xenc = torch::nn::functional::one_hot(xs_tensor, 27).to(torch::kFloat32);

                // 1st param of the trained_params in mapper to the embedding space
                // and 2*n = weights, 2*n + 1 = biases.
                auto emb = torch::matmul(xenc, *trained_params[0]); 
                auto out = emb.view({emb.sizes()[0], emb.sizes()[1] * emb.sizes()[2]});
                for(size_t i=1; i<trained_params.size()-2; i+=2){
                    out = torch::tanh(torch::matmul(out, *trained_params[i]) + (*trained_params[i+1]));
                }
                auto logits = torch::matmul(out, *trained_params[trained_params.size()-2]) + (*trained_params[trained_params.size()-1]);
                auto counts = logits.exp();  // num_names, 27
                auto probs = counts / counts.sum(1, true);

                idx = torch::multinomial(probs, /*num_samples=*/1, /*replacement=*/true, gen).item<int>();
                curr_char = itos[idx];
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
                    const std::vector<torch::Tensor*>& trained_params){
        
        auto xenc = torch::nn::functional::one_hot(validation_data, 27).to(torch::kFloat32);

        // 1st param of the trained_params in mapper to the embedding space
        // and 2*n = weights, 2*n + 1 = biases.
        auto emb = torch::matmul(xenc, *trained_params[0]); 
        auto out = emb.view({emb.sizes()[0], emb.sizes()[1] * emb.sizes()[2]});
        for(size_t i=1; i<trained_params.size()-2; i+=2){
            out = torch::tanh(torch::matmul(out, *trained_params[i]) + (*trained_params[i+1]));
        }
        auto logits = torch::matmul(out, *trained_params[trained_params.size()-2]) + (*trained_params[trained_params.size()-1]);
        auto loss = torch::nn::functional::cross_entropy(logits, validation_target);

        return loss.item<float>();
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


void simple_mlp_model(const std::string& data_path, const int context_win_size, int num_names, int batch_size){

    auto N = torch::ones({27, 27}, torch::kInt32);

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
    
    // xs_tensor shape = [N , context_win_size]
    // ys_tensor shape = [N]

    //________________________________________________________________________________
    // Training data embedding to a lower dimension space
    //________________________________________________________________________________
    auto gen = at::detail::createCPUGenerator(42);
    auto C = torch::randn({27, 2}, gen).set_requires_grad(true); // Embedding space

    // Randomly initialize weights and biases
    auto W1 = torch::randn({2*context_win_size, 100}, gen).set_requires_grad(true);
    auto b1 = torch::randn(100, gen).set_requires_grad(true);
    auto W2 = torch::randn({100, 27}, gen).set_requires_grad(true);
    auto b2 = torch::randn(27, gen).set_requires_grad(true);

    std::vector<torch::Tensor*> params = {&C, &W1, &b1, &W2, &b2};

    // Prior to training, generated name:
    std::cout<<"Generated name(s) prior to training: ";
    std::vector<std::string> gen_names = nn_name_inference(stoi, itos, num_names, context_win_size, params);
    for(int n=0; n<num_names; ++n){
        std::cout<<gen_names[n]<<"   ";
    }
    std::cout<<"\n";
    std::cout<<"________________________________________________________"<<std::endl;

    int iter = 0;
    int iter_10s = -1;
    int num_spacings = 100;
    // Create a linearly spaced tensor from -3 to 0 with 100 elements
    auto lre = torch::linspace(-3, 0, num_spacings);
    auto lrs = torch::pow(10, lre);// 10^-3 ~ 1, exponenetially spaced
    while(true){
        // Mini batch
        auto idx = torch::randint(0, xs_tensor.size(0), {32}, torch::kInt64); // batch size = 32
        auto xs_tensor_batch = xs_tensor.index_select(0, idx);
        auto ys_tensor_batch = ys_tensor.index_select(0, idx);
        auto xenc = torch::nn::functional::one_hot(xs_tensor_batch, 27).to(torch::kFloat32); // shape = [batch size, context_win_size, 27]

        auto emb = torch::matmul(xenc, C); // emb shape = [N, context_win_size, 2]
        //auto emb_unbind = emb.unbind(1); // Unbind into a list of [N, 2] tensors, the list's length = context_win_size
        //auto emb_flattened = torch::cat(emb_unbind, 1); // shape = [N, context_win_size x 2]
        auto emb_flattened = emb.view({emb.sizes()[0], emb.sizes()[1] * emb.sizes()[2]});

        //________________________________________________________________________________
        // First layer
        //________________________________________________________________________________

        auto h = torch::tanh(torch::matmul(emb_flattened, W1) + b1); // shape [N, 100]

        //________________________________________________________________________________
        // Second layer
        //________________________________________________________________________________

        
        auto logits = torch::matmul(h, W2) + b2;

        //________________________________________________________________________________
        // loss calculation
        //________________________________________________________________________________

        auto loss = torch::nn::functional::cross_entropy(logits, ys_tensor_batch);

        if(iter%10==0){
            iter_10s += 1;
            std::cout<<"[Iteration "<<iter<<", Training loss = "<<loss.item<float>()<<"]"<<std::endl;
            std::cout<<"[Evaluation loss = "<<evaluation(xval_tensor, yval_tensor, context_win_size, params)<<"]"<<std::endl;
            std::cout<<"[Inference - simple mlp] Generated name(s): ";
            gen_names.clear();
            gen_names = nn_name_inference(stoi, itos, num_names, context_win_size, params);
            for(int n=0; n<num_names; ++n){
                std::cout<<gen_names[n]<<"   ";
            }
            std::cout<<"\n";
            std::cout<<"________________________________________________________"<<std::endl;
        }
        iter += 1;
        if(iter>1000){
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
        for(size_t i=0; i<params.size(); ++i){
            params[i]->data() -= 0.1*params[i]->grad();
        }

    }
    
   
    return;
}