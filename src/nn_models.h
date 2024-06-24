#pragma once

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>

#include "utils.h"
#include "nn_layers.h"

// Construct a simple neuron of the vocab size weights [27 x 27]
// to learn the bigram model distribution from the input data.
// 
// @param data_path             Path to the input data.
// @param num_names             Number of names to generate at the inference time.
void simple_neuron_model(const std::string& data_path, int num_names);

// Construct a simple multi-layer perceptron (MLP) to learn the distribution
// of the word sequence. Given the context window, it is a next token predictor.
// And until it infers the end-of-word token, it will continously generate the next token
// with the sliding context window.
//
// @param data_path             Path to the input data.
// @param context_win_size      Size of the context window.
// @param num_names             Number of names to generate at the inference time.
// @param batch_size            Size of the batch to be consumed at the training time.
void mlp_model(const std::string& data_path, const int context_win_size, int num_names, int batch_size=64);


// Backprop from scratch - test module
// Manually implementing backprop logic, and compare with the torch grads
//
// @param data_path             Path to the input data.
// @param context_win_size      Size of the context window.
// @param num_names             Number of names to generate at the inference time.
// @param batch_size            Size of the batch to be consumed at the training time.
void custom_backprop_test(const std::string& data_path, const int context_win_size, int num_names, int batch_size=64);

// Backprop from scratch
// MLP with manually implemented backprop logic
//
// @param data_path             Path to the input data.
// @param context_win_size      Size of the context window.
// @param num_names             Number of names to generate at the inference time.
// @param batch_size            Size of the batch to be consumed at the training time.
void custom_backprop_model(const std::string& data_path, const int context_win_size, int num_names, int batch_size=64);