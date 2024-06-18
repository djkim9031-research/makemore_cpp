#include <torch/torch.h>

#include <vector>
#include <string>

#include "utils.h"
#include "bigram_model.h"
#include "nn_models.h"

int main() {
  
    //visualizer("../data/names.txt", N);
    //bigram_model("../data/names.txt", 20);
    //simple_neuron_model("../data/names.txt", 10);
    simple_mlp_model("../data/names.txt", 3, 10);

    return 0;
}