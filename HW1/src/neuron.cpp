#include "neuron.h"

Neuron::Neuron(const vector<double> &initial_weights, const double initial_threshold) {
    weights = initial_weights;
    threshold = initial_threshold;
}

double Neuron::get_output(const vector<double> &input) {
    double sum = 0;
    for (long unsigned int i = 0; i < input.size(); i++) {
        sum += input[i] * weights[i];
    }

    if (sum == threshold) {
        return 1;
    } else return sum > threshold;
};
