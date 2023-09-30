#pragma once

#include "utils.h"

Data read_csv(const std::string &filename);
void normalize_input_data(Data &training_data, Data &data_to_be_normalized);
void shuffle_data(Data &data);
void write_weights_and_biases_to_csv(
    const std::vector<std::vector<double>> &weights_hl,
    const std::vector<double> &weights_ol,
    const std::vector<double> &biases_hl,
    double biases_ol
);