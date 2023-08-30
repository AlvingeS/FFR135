#include<vector>
#include "neuron.h"

#pragma once

using namespace std;

class HopfieldNetwork {
    public:
        HopfieldNetwork(int nr_neurons);
        void train(vector<vector<int>> patterns);
        vector<int> recall(vector<vector<int>> distorted_patterns);
        void print_weights();
        void print_state(int nr_columns);
        void feed_distorted_pattern(vector<int> distorted_pattern);
        void update_neurons();
    private:
        void check_convergence();
        vector<vector<double>> weights;
        vector<Neuron> neurons;
        size_t nr_neurons;
};