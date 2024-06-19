#include "utils.h"

// bigram loss function
// Assumption: Probs of a combination of two chars are independent
// that is, likelihood is simply a product of individual probabilities.
// So, log_likelihood is the sum of individual probabilities.
inline void bigram_loss(std::vector<std::string>& words, 
                        std::unordered_map<char, int>& stoi,
                        torch::Tensor& P){

  float log_likelihood = 0.0;
  int nEvents = 0;

  for(std::string word : words){
    word = "." + word + ".";

    for(size_t i=0; i<word.size()-1; ++i){
      int idx1 = stoi[word[i]];
      int idx2 = stoi[word[i+1]];
      torch::Tensor two_ch_prob = P[idx1][idx2];
      torch::Tensor log_prob = torch::log(two_ch_prob);
      log_likelihood += log_prob.item<float>();
      nEvents += 1;
    }
  }

  float nll = -log_likelihood;
  float normalized_nll = nll/nEvents;

  std::cout<<log_likelihood<<", "<<normalized_nll<<std::endl;

  return;
}

// bigram next token generator.
// `num_name` distinct strings are generated following the bigram model.
inline void bigram_model(std::string data_path, int num_names){

  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto N = torch::ones({27, 27}, options); //Model smoothing with initial count = 1

  std::vector<std::string> words;
  std::unordered_map<int, char> itos;
  std::unordered_map<char, int> stoi;
  tokenizer(data_path, N, words, itos, stoi);

  auto P = N.to(torch::kFloat32);
  P /= P.sum(1, /*keepdim=*/true);

  bigram_loss(words, stoi, P);

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

    std::cout<<"[Inference - Bigram model] Generated name(s): "<<curr_name<<std::endl;
  }

  return;
}
