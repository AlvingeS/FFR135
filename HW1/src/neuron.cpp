#include "neuron.h"
#include <iostream>

Neuron::Neuron(int state, vector_double_ref neuron_weights)
    : state(state), neuron_weights(neuron_weights) {}

void Neuron::update_state(const std::vector<double> input_signals) {
    double sum = 0;
    for (size_t i = 0; i < input_signals.size(); i++) {
        sum += input_signals[i] * neuron_weights[i];
    }

    if (sum == 0) {
        std::cout << "sum is 0" << std::endl;
        set_state(1);
    } else set_state((sum > 0) ? 1 : -1);
};

void Neuron::set_state(int state) {
    this->state = state;
}

int Neuron::get_state() {
    return this->state;
}
