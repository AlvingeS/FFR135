#include "perceptron.h"
#include "neuron.h"
#include <cstddef>
#include <random>
#include <cmath>
#include <bitset>

// Constructor for perceptron
Perceptron::Perceptron(size_t num_inputs)
    : num_inputs(num_inputs),
      weights(num_inputs),
      output_neuron(Neuron(0, &this->weights, 0)) {
    double mean = 0.0;
    double std_dev = 1.0 / sqrt(static_cast<double>(this->num_inputs));

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, std_dev);

    for (size_t i = 0; i < this->num_inputs; ++i) {
        this->weights[i] = distribution(generator);
    }
}

// Trains the perceptron for a given number of epochs
void Perceptron::train(size_t num_epoch, double learning_rate, std::vector<std::vector<int>> all_possible_inputs, std::vector<int> target_values) {
    for (size_t e = 0; e < num_epoch; e++) {
        for (size_t i = 0; i < std::pow(2, this->num_inputs); i++) {
            std::vector<int> input = all_possible_inputs[i];
            this->output_neuron.update_state(input);

            for (size_t j = 0; j < this->num_inputs; j++) {
                this->weights[j] += learning_rate*(target_values[i] - this->output_neuron.get_state())*input[j];
                this->output_neuron.set_bias(this->output_neuron.get_bias() - learning_rate*(target_values[i] - this->output_neuron.get_state()));
            }
        }
    }
}

// Tests if the perceptron is able to separate the input values
bool Perceptron::test_if_separable(std::vector<std::vector<int>> all_possible_inputs, std::vector<int> target_values) {
    for (size_t i = 0; i < std::pow(2, this->num_inputs); i++) {
        std::vector<int> input = all_possible_inputs[i];
        this->output_neuron.update_state(input);

        if (target_values[i] != this->output_neuron.get_state()) {
            return false;
        }
    }

    return true;
}