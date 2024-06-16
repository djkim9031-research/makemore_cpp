#include "utils.h"

// bigram next token generator.
// `num_name` distinct strings are generated following the bigram model.
inline void bigram_model(std::string data_path, int num_names){

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto N = torch::zeros({27, 27}, options);

    std::unordered_map<int, char> itos;
    itos = tokenizer(data_path, N);

    auto P = N.to(torch::kFloat32);
    P = P/P.sum(1, /*keepdim=*/true);

    auto gen = at::detail::createCPUGenerator(42);
    std::vector<std::string> gen_names;
    for(int i=0; i<num_names; ++i){
      std::string curr_name="";
      int idx = 0;
      while (true){
        auto p = P[idx];
        idx = torch::multinomial(p, /*num_samples=*/1, /*replacement=*/true, gen).item<int>();

        if(idx == 0){
          gen_names.push_back(curr_name);
          break;
        }
        curr_name+=itos[idx];
      }

      std::cout<<curr_name<<std::endl;
    }

    return;
}