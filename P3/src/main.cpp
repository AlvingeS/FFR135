#include "network.h"
#include "data_processing.h"
#include <iostream>
#include <vector>
//     Vector<int> hl_dims = {16, 8, 4};
//     network.train(0.005, 0.7, 2, 50, true, true);   
//     hyperfine ./main --runs 3
//     Benchmark #1: No optimization - 9.221s
//     Benchmark #2: Update velocity no longer deletes old matrices and creates new ones 4.606s
//     Benchmark #3: Updated compute_errors to not create new matrices for outer products 3.874s
//     Benchmark #4: Propagate forward does not creat temp vector for compariosn 3.690s
//     Benchmark #5: fixed the last one so outer_prod is not returning anything 3.251s

int main() {
    size_t num_inputs = 2;
    size_t num_outputs = 1;

    Data training_data = read_csv("data/training_set.csv", num_inputs, num_outputs);
    Data validation_data = read_csv("data/validation_set.csv", num_inputs, num_outputs);

    // Normalize the data based on the training data
    normalize_input_data(training_data, training_data);
    normalize_input_data(training_data, validation_data);

    shuffle_data(training_data);

    Vector<int> hl_dims = {16, 8, 4};
    arch_struct arch = {num_inputs, hl_dims, num_outputs};

    // Create and train the network
    Network network(arch, training_data, validation_data);
    network.train(0.005, 0.7, 2, 50, true, true);   

    return 0;
};