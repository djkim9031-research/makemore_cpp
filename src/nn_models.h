#pragma once

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include "utils.h"

void simple_neuron_model(const std::string& data_path, int num_names);

void simple_mlp_model(const std::string& data_path, const int context_win_size, int num_names);