#include "neuron.h"

// Constructor for neuron
Neuron::Neuron(int state, vector_double_ref neuron_weights)
    : state(state), neuron_weights(neuron_weights) {}

// Calculates the new state of the neuron based on the input signals
void Neuron::update_state(const std::vector<double> input_signals) {
    double sum = 0;
    for (size_t i = 0; i < input_signals.size(); i++) {
        sum += input_signals[i] * neuron_weights[i];
    }

    // The state is set to 1 if the sum is geq to 0, otherwise it is set to -1
    if (sum == 0) {
        set_state(1);
    } else set_state((sum > 0) ? 1 : -1);
};

// Setter
void Neuron::set_state(int state) {
    this->state = state;
}

// Getter
int Neuron::get_state() {
    return this->state;
}
