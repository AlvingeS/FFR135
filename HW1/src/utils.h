#pragma once
#include<string>
#include<vector>
#include "neuron.h"

void print_state(size_t num_neurons, size_t num_columns, std::vector<int> states);
std::string convert_for_printing(int state);
std::vector<std::vector<int>> parse_patterns(const std::string& filename);
std::string convert_state_to_scheme(const std::vector<int>& state);
