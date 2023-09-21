#include "neuron.h"
#include <cstddef>
#include <cmath>
#include <iostream>

// Constructor for neuron
Neuron::Neuron(double state, weights_vector_ptr weights, bias_ptr bias)
    : state(state), weights(weights), bias(bias) {}

// Calculates the new state of the neuron based on the input signals
void Neuron::update_state(const std::vector<double> &input_signals) {
    double sum = 0;
    size_t num_inputs = input_signals.size();


    for (size_t i = 0; i < num_inputs; i++) {
        sum += input_signals[i] * (*this->weights)[i];
    }

    this->state = std::tanh(sum - *this->bias);
};