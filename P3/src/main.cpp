#include "network.h"
#include "utils.h"
#include "data_processing.h"
#include <iostream>
#include <vector>


// main function for P2
int main() {

    Data training_data = read_csv("data/training_set.csv");
    Data validation_data = read_csv("data/validation_set.csv");

    normalize_input_data(training_data, training_data);
    normalize_input_data(training_data, validation_data);

    shuffle_data(training_data);

    int_vector nr_neurons = {16, 32, 64};
    double_vector learning_rates = {0.0005, 0.001, 0.005, 0.01};
    double_vector momentums = {0.0, 0.3, 0.6};
    int_vector batch_sizes = {2, 4, 8, 16};
    bool grid_search = false;

    if (grid_search) {
        for (const auto &nr_neuron : nr_neurons) {
            for (const auto &learning_rate : learning_rates) {
                for (const auto &momentum : momentums) {
                    for (const auto &batch_size : batch_sizes) {
                        Network network(nr_neuron, training_data, validation_data);
                        network.train(learning_rate, 0.0001, 0.999, momentum, batch_size, 350, false, false);
                        std::cout << nr_neuron << " " << learning_rate << " " << momentum << " " << batch_size << std::endl;
                    }
                }
            }
        }
    } else {
        Network network(64, training_data, validation_data);
        network.train(0.01, 0.001, 0.999, 0.5, 2, 25000, true, true);
        write_weights_and_biases_to_csv(network.get_weights_hl(), network.get_biases_hl(), network.get_weights_ol(), network.get_biases_ol());      
        network.export_validation_results();
        std::system("python plot.py");
    }

    return 0;
};


/*
C_min: 0.136
32 0.0005 0.3 2
C_min: 0.1364
32 0.0005 0.6 2
C_min: 0.1358
32 0.001 0 2
C_min: 0.1364
32 0.001 0.3 2
32 0.001 0 16
C_min: 0.1364
*/