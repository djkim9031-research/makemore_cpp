#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>

#include <torch/torch.h>
#include <matplotlibcpp.h>

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
inline void visualizer(std::string data_path, torch::Tensor& bigram_tensor){

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