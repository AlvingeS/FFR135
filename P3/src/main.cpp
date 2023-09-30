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

    int_vector nr_neurons = {128};
    double_vector learning_rates = {0.0005, 0.001, 0.005, 0.01};
    double_vector momentums = {0.0, 0.3, 0.6};
    int_vector batch_sizes = {2, 4};
    bool grid_search = false;

    if (grid_search) {
        for (const auto &nr_neuron : nr_neurons) {
            for (const auto &learning_rate : learning_rates) {
                for (const auto &momentum : momentums) {
                    for (const auto &batch_size : batch_sizes) {
                        Network network(nr_neuron, training_data, validation_data);
                        network.train(learning_rate, 0.001, 0.999, momentum, batch_size, 150, false, false, false);
                        std::cout << nr_neuron << " " << learning_rate << " " << momentum << " " << batch_size << std::endl;
                    }
                }
            }
        }
    } else {
        Network network(16, training_data, validation_data);
        network.train(0.01, 0.0005, 0.9999, 0.3, 2, 500, false, true, true);

        write_weights_and_biases_to_csv(network.get_weights_hl(), network.get_weights_ol(), network.get_biases_hl(), network.get_biases_ol());
        network.export_validation_results();
        std::system("python plot.py");
    }

    return 0;
};


// 16 0.01 0.3 2
// 16 0.01 0.3 8
// 16 0.01 0.6 8
// 32 0.005 0.6 8
// 32 0.001 0 8
// 32 0.001 0.6 8


// C_min: 0.1352
// 16 0.01 0.6 2
// C_min: 0.1374
// 32 0.005 0 2
// C_min: 0.1374
// 32 0.01 0 2
// C_min: 0.1366
// 32 0.01 0.6 2
//         network.train(0.00025, 0.00005, 0.9999, 0.9, 2, 50, false, true, true);
