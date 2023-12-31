#include "hopfield_network.h"
#include "utils.h"
#include <iostream>

// Constructor for hopfield network
HopfieldNetwork::HopfieldNetwork()
    : weights(W(num_neurons, std::vector<double>(num_neurons, 0))) {

    // Gives neurons a reference to one vector of weights each
    this->neurons.reserve(num_neurons);
    for (size_t i = 0; i < num_neurons; i++) {
        this->neurons.emplace_back(Neuron(0, this->weights[i]));
    }
}

// Trains the hopfield network on a set of patterns using Hebb's rule
void HopfieldNetwork::train(state_vector patterns) {
    for (size_t i = 0; i < this->num_neurons; i++) {
        for (size_t j = 0; j < this->num_neurons; j++) {
            if (i != j) {
                double sum = 0;

                for (const auto& pattern : patterns) {
                    sum += pattern[i] * pattern[j];
                }

                this->weights[i][j] = sum / this->num_neurons;
            }
        }
    }
}

// Feeds a distorted pattern to the hopfield network by setting the state of the neurons
void HopfieldNetwork::feed_distorted_pattern(state distorted_pattern) {
    for (size_t i = 0; i < this->num_neurons; i++) {
        this->neurons[i].set_state(distorted_pattern[i]);
    }
}

// Updates the state of the neurons in the hopfield network asynchronously
void HopfieldNetwork::update_neurons(bool print) {
    for (size_t i = 0; i < this->num_neurons; i++) {
        std::vector<double> input_signals(this->num_neurons, 0);

        for (size_t j = 0; j < this->num_neurons; j++) {
            input_signals[j] = this->neurons[j].get_state();
        }

        this->neurons[i].update_state(input_signals);

        if (print) {
            

            print_state(this->num_neurons, this->num_columns, get_state());
        }
    }
}

// Returns the state of the neurons in the hopfield network
state HopfieldNetwork::get_state() {
    state state(this->num_neurons, 0);

    for (size_t i = 0; i < this->num_neurons; i++) {
        state[i] = this->neurons[i].get_state();
    }

    return state;
}

// Recalls a pattern from the hopfield network until it reaches a steady state
void HopfieldNetwork::recall(bool print) {
    while (true) {
        state current_state = get_state();
        update_neurons(false);

        if (print) {
            print_state(this->num_neurons, this->num_columns, get_state());
        }

        state new_state = get_state();

        if (calculate_state_differences(current_state, new_state) == 0) {
            break;
        }
    }
}

// Calculates the number of differences between two states
int HopfieldNetwork::calculate_state_differences(const state previous_state, const state current_state) {
    int counter = 0;

    for (size_t i = 0; i < this->num_neurons; i++) {
        if (previous_state[i] != current_state[i]) {
            counter++;
        }
    }

    return counter;
}

// Classifies the a steady state to see if it matches any of the patterns
int HopfieldNetwork::classify_state(state_vector patterns) {
    state state = get_state();

    int counter = 1;

    for (const auto& pattern : patterns) {
        if (state == pattern) {
            return counter;
        }
        counter++;
    }

    return 6;
};