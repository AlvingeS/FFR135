#include "neuron.h"
#include <cstddef>
#include <cmath>
#include <iostream>

// Constructor for neuron
Neuron::Neuron(weights_vector_ptr weights, bias_ptr bias)
    : weights(weights), bias(bias) {}


void Neuron::calculate_net_input(const std::vector<double> &input_signals) {
    double sum = 0;
    size_t num_inputs = input_signals.size();

    for (size_t i = 0; i < num_inputs; i++) {
        sum += input_signals[i] * (*this->weights)[i];
    }

    this->net_input = sum - *this->bias;
}

// Calculates the new state of the neuron based on the input signals
void Neuron::update_state() {
    this->state = std::tanh(this->net_input);
};