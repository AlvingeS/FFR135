#include "perceptron.h"
#include <bitset>
#include <random>
#include <iostream>
#include <unordered_set>

// hyperfine ./main --warmup 2 gave 3.213s on laptop

const size_t n_dim = 5;
const size_t num_samples = 10000;
const size_t num_epoch = 20;
const double learning_rate = 0.05;

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

// main function for P2
int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bitset<32> bitset;

    for (size_t n = 2; n <= n_dim; n++) {
        uint64_t num_possible_functions = std::pow(2, std::pow(2, n));
        uint64_t num_possible_inputs = std::pow(2, n);

        std::vector<int> target_values(num_possible_inputs);
        std::vector<std::vector<int>> all_possible_inputs(num_possible_inputs, std::vector<int>(n));

        std::unordered_set<std::bitset<32>> tested_functions;
        generate_all_possible_inputs(n, num_possible_inputs, all_possible_inputs);

        int num_separable_functions = 0;

        for (size_t i = 0; i < num_samples; i++) {
            Perceptron perceptron(n);
            generate_random_bitset(num_possible_functions, gen, bitset);

            if (tested_functions.find(bitset) != tested_functions.end()) {
                continue; 
            } else {
                tested_functions.insert(bitset);
            }
            
            parse_bitset_to_vector(bitset, num_possible_inputs, target_values);

            perceptron.train(num_possible_inputs, num_epoch, learning_rate, all_possible_inputs, target_values);
            bool function_separable = perceptron.test_if_separable(all_possible_inputs, target_values);

            if (function_separable) {
                num_separable_functions++;
            }

            if (tested_functions.size() == num_possible_functions) {
                break;
            }
        }

        std::cout << "#Separable funcs for n = " << n << ": " << num_separable_functions
                  << " of " << num_possible_functions << " possible functions"
                  << " where " << tested_functions.size() << " where examined " << std::endl;
    }
}