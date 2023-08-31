#include "hopfield_network.h"
#include <unistd.h>
#include "neuron.h"
#include <iostream>

HopfieldNetwork::HopfieldNetwork(int num_neurons)
    : num_neurons(num_neurons), weights(vector2d_double(num_neurons, std::vector<double>(num_neurons, 0))) {

    this->neurons.reserve(num_neurons);
    for (size_t i = 0; i < num_neurons; i++) {
        this->neurons.emplace_back(Neuron(0, this->weights[i]));
    }
}

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

void HopfieldNetwork::feed_distorted_pattern(std::vector<int> distorted_pattern) {
    for (size_t i = 0; i < this->num_neurons; i++) {
        this->neurons[i].set_state(distorted_pattern[i]);
    }
}

void HopfieldNetwork::update_neurons() {
    for (size_t i = 0; i < this->num_neurons; i++) {
        std::vector<double> input_signals(this->num_neurons, 0);

        for (size_t j = 0; j < this->num_neurons; j++) {
            input_signals[j] = this->neurons[j].get_state();
        }

        this->neurons[i].update_state(input_signals);

        print_state(10);
    }
}

const void HopfieldNetwork::print_weights() {
    for (const auto& row : this->weights) {
        for (const auto& weight : row) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
    }
}

std::string HopfieldNetwork::convert_for_printing(int state) {
    if (state == 1) {
        return "X";
    } else {
        return " ";
    }
}

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
    
    // Optional: add a short sleep to make the evolution easier to follow
    usleep(10000);  // Sleep for 500,000 microseconds (0.5 seconds)
}