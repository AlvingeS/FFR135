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

    int_vector nr_neurons = {16, 32};
    double_vector learning_rates = {0.01, 0.0075, 0.005, 0.0025};
    double_vector momentums = {0.0, 0.25, 0.5};
    int_vector batch_sizes = {1};
    bool grid_search = true;

    if (grid_search) {
        for (const auto &nr_neuron : nr_neurons) {
            for (const auto &learning_rate : learning_rates) {
                for (const auto &momentum : momentums) {
                    for (const auto &batch_size : batch_sizes) {

                        Network network(nr_neuron, training_data, validation_data);
                        network.train(learning_rate, momentum, batch_size, 350, true, false, false);
                        std::cout << nr_neuron << " " << learning_rate << " " << momentum << " " << batch_size << std::endl;
                    }
                }
            }
        }
    } else {
        Network network(16, training_data, validation_data);
        network.train(0.005, 0.5, 32, 350, true, false, true);
    }

    return 0;
};