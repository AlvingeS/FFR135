#include "perceptron.h"
#include "utils.h"
#include <iostream>
#include <unordered_set>

// hyperfine ./main --warmup 2 gave 3.213s on laptop

const size_t n_dim = 5;
const size_t num_samples = 10000;
const size_t num_epoch = 20;
const double learning_rate = 0.05;

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