#pragma once

#include "utils.h"

Data read_csv(const std::string &filename);
void normalize_input_data(Data &training_data, Data &data_to_be_normalized);
void shuffle_data(Data &data);
