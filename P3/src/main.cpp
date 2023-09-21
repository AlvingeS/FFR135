#include "network.h"
#include "utils.h"
#include <iostream>
#include <vector>

// main function for P2
int main() {

    auto csv_data = read_csv("training_set.csv");
    auto csv_validation_data = read_csv("validation_set.csv");

    auto normalized_data = normalize_data(csv_data.first);
    auto normalized_validation_data = normalize_data(csv_validation_data.first);



    Network network(3, csv_data.first.size(), csv_validation_data.first.size());

    network.set_input_patterns(normalized_data);
    network.set_targets(csv_data.second);
    network.set_validation_input_patterns(normalized_validation_data);
    network.set_validation_targets(csv_validation_data.second);

    network.train(0.0001, 250, 1000);

    return 0;
}