#include "hopfield_network.h"

#include "neuron.h"
#include <iostream>

HopfieldNetwork::HopfieldNetwork(int nr_neurons)
    : nr_neurons(nr_neurons) {
    this->weights = vector<vector<double>>(nr_neurons, vector<double>(nr_neurons, 0));
    this->neurons = vector<vector<Neuron>>(nr_neurons, vector<Neuron>(nr_neurons, Neuron(this->weights[0], 0)));
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

void HopfieldNetwork::print_weights() {
    for (const auto& row : this->weights) {
        for (const auto& weight : row) {
            cout << weight << " ";
        }
        cout << endl;
    }
}