#include <torch/torch.h>

#include <vector>
#include <string>

#include "utils.h"
#include "bigram_model.h"
#include "nn_models.h"

int main() {
  
    //viz_bigram("../data/names.txt", N);
    //bigram_model("../data/names.txt", 10);
    //simple_neuron_model("../data/names.txt", 10);

    //mlp_model("../data/names.txt", 5, 10);
    mlp_model_with_custom_backprop("../data/names.txt", 5, 10);

    return 0;
}