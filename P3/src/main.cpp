#include "network.h"
#include "data_processing.h"
#include <iostream>
#include <vector>

int main() {
    size_t num_inputs = 2;
    size_t num_outputs = 1;

    Data training_data = read_csv("data/training_set.csv", num_inputs, num_outputs);
    Data validation_data = read_csv("data/validation_set.csv", num_inputs, num_outputs);

    // Normalize the data based on the training data
    normalize_input_data(training_data, training_data);
    normalize_input_data(training_data, validation_data);

    shuffle_data(training_data);

    Vector<int> hl_dims = {32, 16, 8};
    arch_struct arch = {num_inputs, hl_dims, num_outputs};

    // Create and train the network
    Network network(arch, training_data, validation_data);
    network.train(0.005, 0.7, 2, 1000, true, true);   

    return 0;
};