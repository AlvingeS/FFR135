#pragma once

#include "neuron.h"
#include <cstddef>
#include <vector>
#include <cstdint>

typedef std::vector<double> weights_vector;

class Perceptron {
    public:
        Perceptron(size_t num_inputs);
        void train(uint64_t num_possible_inputs, size_t num_epoch, double learning_rate, std::vector<std::vector<int>> &all_possible_values, std::vector<int> &target_values);
        bool test_if_separable(std::vector<std::vector<int>> &all_possible_inputs, std::vector<int> &target_values);
    private:
        size_t num_inputs;
        weights_vector weights;
        Neuron output_neuron;
};