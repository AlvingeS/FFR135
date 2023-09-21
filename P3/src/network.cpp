#include "network.h"
#include "neuron.h"
#include <random>
#include <cmath>
#include <iostream>

// Constructor for network
Network::Network(size_t num_hl_neurons)
    : num_hl_neurons(num_hl_neurons),
      hl_biases(num_hl_neurons),
      ol_bias(0.0),
      ol_neuron(0, &this->ol_weights, &this->ol_bias) {
    
    double mean = 0.0;
    double std_dev = 1.0 / sqrt(static_cast<double>(this->num_inputs));
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, std_dev);

    this->hl_weights = weights_matrix(this->num_hl_neurons, weights_vector(2));
    this->ol_weights = weights_vector(this->num_hl_neurons);

    this->hl_neurons.reserve(this->num_hl_neurons);

    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        this->hl_weights[i][0] = distribution(generator);
        this->hl_weights[i][1] = distribution(generator);
        this->ol_weights[i] = distribution(generator);

        this->hl_neurons.emplace_back(0, &this->hl_weights[i], &this->hl_biases[i]);
    }
};


void Network::forward_call(const std::vector<double> &input_signals) {
    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        this->hl_neurons[i].update_state(input_signals);
    }

    this->ol_neuron.update_state(this->get_hl_states());
};

std::vector<double> Network::get_hl_states() {
    std::vector<double> hl_states(this->num_hl_neurons);

    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        hl_states[i] = this->hl_neurons[i].get_state();
    }

    return hl_states;
};