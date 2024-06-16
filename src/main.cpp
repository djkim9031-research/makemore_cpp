#include <torch/torch.h>

#include <vector>
#include <string>

#include "utils.h"
#include "bigram_model.h"

int main() {
  
    //visualizer("../data/names.txt", N);
    bigram_model("../data/names.txt", 20);

    return 0;
}