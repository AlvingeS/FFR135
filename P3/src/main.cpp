#include "network.h"
#include "utils.h"
#include "data_processing.h"
#include <iostream>
#include <vector>

int main() {
    Data training_data = read_csv("data/training_set.csv");
    Data validation_data = read_csv("data/validation_set.csv");

    // Normalize the data based on the training data
    normalize_input_data(training_data, training_data);
    normalize_input_data(training_data, validation_data);

    shuffle_data(training_data);

    // Create and train the network
    Network network(32, training_data, validation_data);
    network.train(0.005, 0.7, 2, 500, true, true);
    parameters_struct params = network.get_parameters();
    write_weights_and_biases_to_csv(params.hl_w, params.hl_b, params.ol_w, params.ol_b);      


    return 0;
};