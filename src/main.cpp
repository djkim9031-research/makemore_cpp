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

    mlp_model("../data/names.txt", 8, 10);
    //custom_backprop_test("../data/names.txt", 5, 10);
    //custom_backprop_model("../data/names.txt", 5, 10);

    return 0;
}