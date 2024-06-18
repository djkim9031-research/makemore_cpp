#include "nn_models.h"

namespace{
    std::vector<std::string> simple_name_generator(std::unordered_map<char, int>& stoi,
                                                   std::unordered_map<int, char>& itos, 
                                                   const torch::Tensor& trained_weights,
                                                   int num_names){
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
    std::cout<<"Generated name prior to training: ";
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
            std::cout<<"[Inference] Generated name: ";
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


void simple_mlp_model(const std::string& data_path, const int context_win_size, int num_names){

    auto N = torch::ones({27, 27}, torch::kInt32);

    std::vector<std::string> words;
    std::unordered_map<int, char> itos;
    std::unordered_map<char, int> stoi;
    tokenizer(data_path, N, words, itos, stoi);

    //________________________________________________________________________________
    // Data preparation and create training data
    //________________________________________________________________________________

    std::vector<int64_t> xs; // num_existing_char x context_win_size
    std::vector<int64_t> ys; // num_exisiting_char
    for(const std::string& word : words){
        std::vector<int> context;
        for(size_t i=0; i<context_win_size; ++i){
            context.push_back(0);
        }

        std::string curr_name = word + ".";
        for(char ch : curr_name){
            int idx = stoi[ch];
            xs.insert(xs.end(), context.begin(), context.end());
            ys.push_back(idx);

            for(int i=0; i<context_win_size - 1; ++i){
                context[i] = context[i+1];
            }
            context[context_win_size - 1] = idx;

        }
    }

    // Convert to torch tensors.
    auto xs_tensor = torch::from_blob(xs.data(), {static_cast<int>(ys.size()), context_win_size}, torch::kInt64).clone();
    auto ys_tensor = torch::tensor(ys, torch::kInt64);

    // xs_tensor shape = [N , context_win_size]
    // ys_tensor shape = [N]

    //________________________________________________________________________________
    // Training data embedding to a lower dimension space
    //________________________________________________________________________________
    auto gen = at::detail::createCPUGenerator(42);
    auto xenc = torch::nn::functional::one_hot(xs_tensor, 27).to(torch::kFloat32); // shape = [N, context_win_size, 27]
    auto C = torch::randn({27, 2}, gen).set_requires_grad(true); // Embedding space

    // Randomly initialize weights and biases
    auto W1 = torch::randn({2*context_win_size, 100}, gen).set_requires_grad(true);
    auto b1 = torch::randn(100, gen).set_requires_grad(true);
    auto W2 = torch::randn({100, 27}, gen).set_requires_grad(true);
    auto b2 = torch::randn(27, gen).set_requires_grad(true);

    std::vector<torch::Tensor*> params = {&C, &W1, &b1, &W2, &b2};

    int iter = 0;
    while(true){
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

        auto loss = torch::nn::functional::cross_entropy(logits, ys_tensor);

        if(iter%10==0){
            std::cout<<"[Iteration "<<iter<<", loss = "<<loss.item<float>()<<"]"<<std::endl;
            std::cout<<"\n";
            std::cout<<"________________________________________________________"<<std::endl;
        }
        iter += 1;
        if(loss.item<float>() < 2.47){ //Roughly the nll loss of bigram
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
        for(size_t i=0; i<params.size(); ++i){
            params[i]->data() -= 0.1*params[i]->grad();
        }

    }
    
   
    return;
}