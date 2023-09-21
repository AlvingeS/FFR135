#include "network.h"
#include "utils.h"
#include <iostream>
#include <vector>

// main function for P2
int main() {

    auto csv_data = read_csv("training_set.csv");
    auto csv_validation_data = read_csv("validation_set.csv");

    Network network(16, csv_data.first.size(), csv_validation_data.first.size());

    network.set_input_patterns(csv_data.first);
    network.set_targets(csv_data.second);
    network.set_validation_input_patterns(csv_validation_data.first);
    network.set_validation_targets(csv_validation_data.second);

    network.train(0.001, 100, 1000);

    return 0;
}