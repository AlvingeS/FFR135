#include "network.h"
#include "utils.h"
#include "data_processing.h"
#include <iostream>
#include <vector>

// main function for P2
int main() {

    Data training_data = read_csv("training_set.csv");
    Data validation_data = read_csv("validation_set.csv");

    normalize_input_data(training_data);
    normalize_input_data(validation_data);

    // print_matrix(training_data.inputs);
    // print_vector(training_data.targets);

    int_vector num_hl_neurons = {8, 16, 32};
    double_vector learning_rates = {0.01, 0.005, 0.001, 0.0005};
    int_vector batch_sizes = {1};
    double_vector momentums = {0, 0.2};

    for (size_t i = 0; i < num_hl_neurons.size(); i++) {
        for (size_t j = 0; j < learning_rates.size(); j++) {
            for (size_t k = 0; k < batch_sizes.size(); k++) {
                for (size_t l = 0; l < momentums.size(); l++) {
                    // Prints all data in one line
                    std::cout << num_hl_neurons[i] << " " << learning_rates[j] << " " << batch_sizes[k] << " " << momentums[l] << " ";
                    Network network(num_hl_neurons[i], training_data, validation_data);
                    network.train(learning_rates[j], momentums[l], batch_sizes[k], 300, true);
                }
            }
        }
    }

    // 16 0.005 4 0.5 C_min: 0.1282
    // 32 0.01 4 0 C_min: 0.1282
    // 32 0.005 4 0 C_min: 0.1304
    // 16 0.01 2 0 C_min: 0.1282
    // 32 0.005 2 0.1 C_min: 0.124

    // 32 0.01 1 0 C_min: 0.1206 SGD
    // 16 0.005 1 0 C_min: 0.1196 SGD
    // 32 0.01 1 0 C_min: 0.1192 SGD
    // 32 0.01 1 0 C_min: 0.1196 SGD




    return 0;
};