#include "neuron.h"
#include <cstddef>

// Constructor for neuron
Neuron::Neuron(int state, vector_double_ptr neuron_weights, double bias)
    : state(state), neuron_weights_ptr(neuron_weights), bias(bias) {}

// Calculates the new state of the neuron based on the input signals
void Neuron::update_state(std::vector<int> &input_signals) {
    double sum = 0;
    for (size_t i = 0; i < input_signals.size(); i++) {
        sum += input_signals[i] * (*neuron_weights_ptr)[i];
    }

    // The state is set to 1 if the sum is geq to bias, otherwise set to 0
    if (sum >= this->bias) {
        set_state(1);
    } else set_state(-1);
};
