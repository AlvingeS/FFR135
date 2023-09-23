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

    Network network(8, training_data, validation_data);

    network.train(0.0001, 250, 1000);

    return 0;
};