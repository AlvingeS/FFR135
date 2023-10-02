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

    int_vector nr_neurons = {8, 16, 32, 64};
    double_vector learning_rates = {0.005, 0.01};
    double_vector momentums = {0.7};
    int_vector batch_sizes = {2};

    Network network(32, training_data, validation_data);
    network.train(0.005, 0.0001, 1, 0.7, 2, 10000, true, true);
    write_weights_and_biases_to_csv(network.get_weights_hl(), network.get_biases_hl(), network.get_weights_ol(), network.get_biases_ol());      
    network.export_validation_results();
    std::system("python plot.py");

    return 0;
};


/*

16 0.005, no decay, 0.9, 4 and 500 epochs gave me triangle shape


H_min: 2134.4
16 0.005 0.7 2

H_min: 2099.97
32 0.005 0.7 2



*/