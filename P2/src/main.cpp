#include "perceptron.h"
#include <bitset>
#include <random>
#include <iostream>

const size_t n_dim = 5;
const size_t num_samples = 10000;
const size_t num_epoch = 20;
const double learning_rate = 0.05;

std::vector<std::vector<int>> generate_all_possible_inputs(int n) {
    std::vector<std::vector<int>> all_possible_inputs(std::pow(2, n), std::vector<int>(n));

    for (size_t i = 0; i < std::pow(2, n); i++) {
        std::bitset<8> bits(i);

        for (size_t j = 0; j < n; j++) {
            all_possible_inputs[i][j] = bits[j];
        }
    }

    return all_possible_inputs;
}

// Generates the binary target values for the perceptron
std::vector<int> generate_random_target_values(int n) {
    std::vector<int> target_values(std::pow(2, n));

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> distr(0, 1);

    for (size_t i = 0; i < std::pow(2, n); i++) {
        target_values[i] = distr(gen);
    }

    return target_values;
}  

// main function for P2
int main() {
    for (size_t n = 2; n < n_dim + 1; n++) {

        std::vector<std::vector<int>> all_possible_inputs = generate_all_possible_inputs(n);
        std::vector<int> target_values = generate_random_target_values(n);

        Perceptron perceptron(n);
        int num_separable_functions = 0;

        for (size_t i = 0; i < num_samples; i++) {
            perceptron.train(num_epoch, learning_rate, all_possible_inputs, target_values);
            bool function_separable = perceptron.test_if_separable(all_possible_inputs, target_values);

            if (function_separable) {
                num_separable_functions++;
                break;
            }
        }

        std::cout << "Number of separable functions for n = " << n << ": " << num_separable_functions << std::endl;
    }
}