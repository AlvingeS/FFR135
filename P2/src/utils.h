#pragma once

#include <cstddef>
#include <vector>
#include <bitset>
#include <random>
#include <cstdint>

void generate_all_possible_inputs(size_t n, uint64_t num_possible_inputs, std::vector<std::vector<int>>& all_possible_inputs);
void generate_random_bitset(uint64_t num_possible_functions, std::mt19937 &gen, std::bitset<32> &bitset);
void parse_bitset_to_vector(std::bitset<32> &bitset, uint64_t num_possible_inputs, std::vector<int>& target_values);
void print_matrix(std::vector<std::vector<int>> &matrix);
void print_vector(std::vector<int> &vector);

