#include "hopfield_network.h"
#include <unistd.h>
#include "neuron.h"
#include <iostream>

// Constructor for hopfield network
HopfieldNetwork::HopfieldNetwork(int num_neurons)
    : num_neurons(num_neurons), weights(vector2d_double(num_neurons, std::vector<double>(num_neurons, 0))) {

    // Gives neurons a reference to one vector of weights each
    this->neurons.reserve(num_neurons);
    for (size_t i = 0; i < num_neurons; i++) {
        this->neurons.emplace_back(Neuron(0, this->weights[i]));
    }
}

// Trains the hopfield network on a set of patterns using Hebb's rule
void HopfieldNetwork::train(vector2d_int patterns) {
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

std::vector<int> HopfieldNetwork::recall(vector2d_int distorted_patterns) {
    std::vector<int> classified_numbers(distorted_patterns.size(), 0);
    return classified_numbers;
}

// Feeds a distorted pattern to the hopfield network by setting the state of the neurons
void HopfieldNetwork::feed_distorted_pattern(std::vector<int> distorted_pattern) {
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
            print_state(10);
        }
    }
}

// Returns the state of the neurons in the hopfield network
std::vector<int> HopfieldNetwork::get_state() {
    std::vector<int> state(this->num_neurons, 0);

    for (size_t i = 0; i < this->num_neurons; i++) {
        state[i] = this->neurons[i].get_state();
    }

    return state;
}

// Classifies the a steady state to see if it matches any of the patterns
int HopfieldNetwork::classify_state(vector2d_int patterns) {
    std::vector<int> state = get_state();

    int counter = 1;

    for (const auto& pattern : patterns) {
        if (state == pattern) {
            return counter;
        }
        counter++;
    }

    return 6;
}

// Prints the weights of the hopfield network (For debugging purposes)
const void HopfieldNetwork::print_weights() {
    for (const auto& row : this->weights) {
        for (const auto& weight : row) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
    }
}

// Converts a state to an easy-to-read string
std::string HopfieldNetwork::convert_for_printing(int state) {
    if (state == 1) {
        return "X";
    } else {
        return " ";
    }
}

// Prints the state of the neurons in the hopfield network
// Flushes the screen to see the evolution of the state
const void HopfieldNetwork::print_state(int nr_columns) {
    std::cout << "\033[2J\033[1;1H";  // Clear screen and move cursor to top-left corner

    int counter = 0;

    for (size_t i = 0; i < this->num_neurons; i++) {
        std::cout << convert_for_printing(this->neurons[i].get_state()) << " ";

        if (++counter % nr_columns == 0) {
            std::cout << std::endl;
            counter = 0;
        }
    }
    std::cout << std::endl;
    
    usleep(5000);
}