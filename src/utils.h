#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <cassert>
#include <map>

#include <torch/torch.h>
#include <matplotlibcpp.h>

#include "nn_layers.h"

// Custom hash function for std::pair<char, char>
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
    }
};

// Function to read lines from a file into a vector
inline std::vector<std::string> readLinesFromFile(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return lines;
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}

// bigram tokenizer
inline void tokenizer(std::string data_path, 
                      torch::Tensor& bigram_tensor,
                      std::vector<std::string>& words, 
                      std::unordered_map<int, char> &itos, 
                      std::unordered_map<char, int> &stoi){

    // Read words from file
    words = readLinesFromFile(data_path);

    std::unordered_map<std::pair<char, char>, int, PairHash> b;
    std::set<char> unique_chars;

    for(std::string word : words){
      for(char ch : word){
        unique_chars.insert(ch);
      }
      word = "." + word + ".";

      for(size_t i=0; i<word.size()-1; ++i){
        std::pair<char, char> key = std::make_pair(word[i], word[i+1]);
        b[key]++;
      }
    }

    std::vector<std::pair<std::pair<char, char>, int>> vec(b.begin(), b.end());
    std::vector<char> sorted_chars(unique_chars.begin(), unique_chars.end());
    std::sort(sorted_chars.begin(), sorted_chars.end());
    
    std::sort(vec.begin(), vec.end(), [](const auto& lhs, const auto& rhs){
      return lhs.second > rhs.second;
    });

    for (size_t i = 0; i < sorted_chars.size(); ++i) {
        stoi[sorted_chars[i]] = static_cast<int>(i) + 1;
        itos[static_cast<int>(i) + 1] = sorted_chars[i];
    }
    stoi['.'] = 0;
    itos[0] = '.';

     for (const auto& pair : b) {
        char ch1 = pair.first.first;
        char ch2 = pair.first.second;
        int idx1 = stoi[ch1];
        int idx2 = stoi[ch2];
        bigram_tensor[idx1][idx2] += pair.second;
    }

    return;
}

// Bigram visualizer
namespace plt = matplotlibcpp;
inline void viz_bigram(std::string data_path, torch::Tensor& bigram_tensor){

    std::vector<std::string> words;
    std::unordered_map<int, char> itos;
    std::unordered_map<char, int> stoi;
    tokenizer(data_path, bigram_tensor, words, itos, stoi);
    int rows = bigram_tensor.size(0);
    int cols = bigram_tensor.size(1);

    plt::figure_size(1600, 1600);
    
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            std::string chstr = std::string(1, itos[i]) + std::string(1, itos[j]);
            plt::text((double)j, cols - (double)i, chstr);
            plt::text((double)j, cols - 0.5 -(double)i, std::to_string(bigram_tensor[i][j].item<int>()));
        }
    }
    plt::xlim(0.0, (double)cols);
    plt::ylim(0.0, (double)rows);
    plt::axis("off");
    plt::save("bigram_plot.png");

    return;
}

// Visualizer to see what embedding space has learned.
// Only accepts 2D embedding space.
inline void viz_embedding_space(const torch::Tensor& embedding, const std::unordered_map<int, char>& itos){
    assert(embedding.size(1) == 2); // Only able to visualize 2-dim, hence 2-dim embedding space is supported.
    plt::figure_size(1600, 1600);

    std::vector<float> x(embedding.size(0));
    std::vector<float> y(embedding.size(0));
    for (int i = 0; i < embedding.size(0); ++i) {
        x[i] = embedding[i][0].item<float>();
        y[i] = embedding[i][1].item<float>();
    }
    plt::scatter(x, y, 200);
    for (int i = 0; i < embedding.size(0); ++i) {
        plt::text(x[i], y[i], std::string(1, itos.at(i)));
    }

    plt::grid(true);
    plt::save("embedding_space_plot.png");

}

// Visualizer helper function to see the tanh layer's output distribution.
// This is to check if the initialization of weights are valid, and not leading to value
// saturation by the nonlinearity function, i.e., tanh.
inline void viz_tanh_activation_dist(const std::vector<std::unique_ptr<Layer>>& layers) {
    plt::figure_size(2000, 400);

    for (size_t i = 0; i < layers.size() - 1; ++i) { // Exclude the output layer
        if(layers[i]->name().find("tanh") != std::string::npos){
            auto tanh_layer = dynamic_cast<TanhActivation*>(layers[i].get());
            if(tanh_layer){
                torch::Tensor t = tanh_layer->output;
                float mean = t.mean().item<float>();
                float std = t.std().item<float>();
                float saturated = (t.abs() > 0.97).to(torch::kFloat32).mean().item<float>() * 100;

                std::cout << "layer " << i << " (" << tanh_layer->name() << "): mean " << mean << ", std " << std
                          << ", saturated: " << saturated << "%" << std::endl;

                auto hist = torch::histc(t, /*bins=*/100);
                auto edges = torch::linspace(-1, 1, 101); // 100 bins have 101 edges
                std::vector<float> hy(hist.data_ptr<float>(), hist.data_ptr<float>() + hist.numel());
                std::vector<float> hx(edges.data_ptr<float>(), edges.data_ptr<float>() + edges.numel() - 1);

                plt::plot(hx, hy, {{"label", "layer " + std::to_string(i) + " (" + tanh_layer->name() + ")"}});
            }
        }
    }

    plt::legend();
    plt::title("activation distribution");
    plt::save("tanh_activation.png");
}


// Compare the implemented gradients to the Torch gradients.
inline void cmp(const std::string& s, const torch::Tensor& dt, const torch::Tensor &t){
    // Check for exact equality
    bool exact = torch::all(dt == t.grad()).item<bool>();

    // Check for approximate equality
    bool approximate = torch::allclose(dt, t.grad());

    // Calculate the maximum difference
    float maxdiff = (dt - t.grad()).abs().max().item<float>();


    std::cout << std::setw(15) << s << " | exact: " << std::boolalpha << exact
              << " | approximate: " << approximate << " | maxdiff: " << maxdiff << std::endl;
}
