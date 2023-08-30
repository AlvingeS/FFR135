#include "hopfield_network.h"
#include <unistd.h>
#include "neuron.h"
#include <iostream>

HopfieldNetwork::HopfieldNetwork(int nr_neurons)
    : nr_neurons(nr_neurons) {
    this->weights = vector<vector<double>>(nr_neurons, vector<double>(nr_neurons, 0));

    for (size_t i = 0; i < nr_neurons; i++) {
        this->neurons.push_back(Neuron(0, this->weights[i]));
    }
}

void HopfieldNetwork::train(vector<vector<int>> patterns) {
    for (size_t i = 0; i < this->nr_neurons; i++) {
        for (size_t j = 0; j < this->nr_neurons; j++) {
            if (i != j) {
                double sum = 0;

                for (const auto& pattern : patterns) {
                    sum += pattern[i] * pattern[j];
                }

                this->weights[i][j] = sum / this->nr_neurons;
            }
        }
    }
}

vector<int> HopfieldNetwork::recall(vector<vector<int>> distorted_patterns) {
    vector<int> classified_numbers(distorted_patterns.size(), 0);
    return classified_numbers;
}

void HopfieldNetwork::feed_distorted_pattern(vector<int> distorted_pattern) {
    for (size_t i = 0; i < this->nr_neurons; i++) {
        this->neurons[i].set_state(distorted_pattern[i]);
    }
}

void HopfieldNetwork::update_neurons() {
    for (size_t i = 0; i < this->nr_neurons; i++) {
        vector<double> input_signals(this->nr_neurons, 0);

        for (size_t j = 0; j < this->nr_neurons; j++) {
            input_signals[j] = this->neurons[j].get_state();
        }

        this->neurons[i].update_state(input_signals);

        print_state(10);
    }
}

void HopfieldNetwork::print_weights() {
    for (const auto& row : this->weights) {
        for (const auto& weight : row) {
            cout << weight << " ";
        }
        cout << endl;
    }
}

string convert_for_printing(int state) {
    if (state == 1) {
        return "X";
    } else {
        return " ";
    }
}

void HopfieldNetwork::print_state(int nr_columns) {
    cout << "\033[2J\033[1;1H";  // Clear screen and move cursor to top-left corner

    int counter = 0;

    for (size_t i = 0; i < this->nr_neurons; i++) {
        cout << convert_for_printing(this->neurons[i].get_state()) << " ";

        if (++counter % nr_columns == 0) {
            cout << endl;
            counter = 0;
        }
    }
    cout << endl;
    
    // Optional: add a short sleep to make the evolution easier to follow
    usleep(25000);  // Sleep for 500,000 microseconds (0.5 seconds)
}