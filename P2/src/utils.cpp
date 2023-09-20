#include "utils.h"
#include <iostream>

void generate_all_possible_inputs(size_t n, uint64_t num_possible_inputs, std::vector<std::vector<int>>& all_possible_inputs) {
    for (size_t i = 0; i < num_possible_inputs; i++) {
        std::bitset<32> bits(i);

        for (size_t j = 0; j < n; j++) {
            all_possible_inputs[i][j] = bits[j];
        }
    }
}

void generate_random_bitset(uint64_t num_possible_functions, std::mt19937 &gen, std::bitset<32> &bitset) {
    std::uniform_int_distribution<uint64_t> dis(0, num_possible_functions - 1);

    bitset = std::bitset<32>(dis(gen));
}

 void parse_bitset_to_vector(std::bitset<32> &bitset, uint64_t num_possible_inputs, std::vector<int>& target_values) {
    for (size_t i = 0; i < num_possible_inputs; i++) {
        target_values[i] = bitset[i] == 1 ? 1 : -1;
    }
}

void print_matrix(std::vector<std::vector<int>> &matrix) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[i].size(); j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void print_vector(std::vector<int> &vector) {
    for (size_t i = 0; i < vector.size(); i++) {
        std::cout << vector[i] << " ";
    }

    std::cout << std::endl;
}