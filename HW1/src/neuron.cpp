#include "neuron.h"

Neuron::Neuron(vector<double>& neuron_weights, double threshold)
    : neuron_weights(neuron_weights), threshold(threshold) {
}

void Neuron::update_state(const vector<double> input_signals) {
    double sum = 0;
    for (size_t i = 0; i < input_signals.size(); i++) {
        sum += input_signals[i] * neuron_weights[i];
    }

    if (sum == this->threshold) {
        this->state = 1;
    } else this->state = (sum > threshold) ? 1 : -1;
};
