#include "network.h"
#include "utils.h"
#include "data_processing.h"
#include <iostream>
#include <vector>

// main function for P2
int main() {

    Data training_data = read_csv("training_set.csv");
    Data validation_data = read_csv("validation_set.csv");

    normalize_input_data(training_data, training_data);
    normalize_input_data(training_data, validation_data);

    shuffle_data(training_data);

    Network network(16, training_data, validation_data);

    int_vector nr_neurons = {32};
    double_vector learning_rates = {0.005};
    double_vector momentums = {0.9};
    int_vector batch_sizes = {8};

    for (const auto &nr_neuron : nr_neurons) {
        for (const auto &learning_rate : learning_rates) {
            for (const auto &momentum : momentums) {
                for (const auto &batch_size : batch_sizes) {
                    // prints all parameters in one line
                    Network network(nr_neuron, training_data, validation_data);
                    network.train(learning_rate, momentum, batch_size, 250, false);
                    std::cout << nr_neuron << " " << learning_rate << " " << momentum << " " << batch_size << std::endl;
                }
            }
        }
    }

    return 0;
};