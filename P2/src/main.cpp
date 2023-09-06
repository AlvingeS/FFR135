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

std::vector<std::vector<int>> generate_all_possible_inputs(int n) {
    std::vector<std::vector<int>> all_possible_inputs(std::pow(2, n), std::vector<int>(n));

    for (size_t i = 0; i < std::pow(2, n); i++) {
        std::bitset<32> bits(i);

        for (size_t j = 0; j < n; j++) {
            all_possible_inputs[i][j] = bits[j];
        }
    }

    return all_possible_inputs;
}

// Generates the binary target values for the perceptron
std::bitset<32> generate_random_bitset(int n, std::mt19937 &gen) {
    std::uniform_int_distribution<uint64_t> dis(0, std::pow(2, std::pow(2, n)) - 1);

    return std::bitset<32>(dis(gen));
}

std::vector<int> parse_bitset_to_vector(std::bitset<32> bitset, int n) {
    std::vector<int> target_values(std::pow(2, n));

    for (size_t i = 0; i < std::pow(2, n); i++) {
        target_values[i] = bitset[i] == 1 ? 1 : -1;
    }

    return target_values;
}

void print_matrix(std::vector<std::vector<int>> matrix) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[i].size(); j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

// main function for P2
int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t n = 2; n <= n_dim; n++) {

        std::unordered_set<std::bitset<32>> *tested_functions = new std::unordered_set<std::bitset<32>>();
        std::vector<std::vector<int>> *all_possible_inputs = new std::vector<std::vector<int>>(generate_all_possible_inputs(n));

        // print_matrix(all_possible_inputs);
        // print_matrix(std::vector<std::vector<int>>({target_values}));

        int num_separable_functions = 0;

        for (size_t i = 0; i < num_samples; i++) {
            Perceptron perceptron(n);
            std::bitset<32> bitset = generate_random_bitset(n, gen);

            if (tested_functions->find(bitset) != tested_functions->end()) {
                continue; 
            } else {
                tested_functions->insert(bitset);
            }

            std::vector<int> target_values = parse_bitset_to_vector(bitset, n);

            perceptron.train(num_epoch, learning_rate, *all_possible_inputs, target_values);
            bool function_separable = perceptron.test_if_separable(*all_possible_inputs, target_values);

            if (function_separable) {
                num_separable_functions++;
            }

            if (tested_functions->size() == std::pow(2, std::pow(2, n))) {
                break;
            }
        }

        std::cout << "# Separable funcs for n = " << n << ": " << num_separable_functions
                  << " of " << std::pow(2, std::pow(2, n)) << " possible functions"
                  << " where " << tested_functions->size() << " where examined " << std::endl;
       
        delete tested_functions;
        delete all_possible_inputs;
    }
}