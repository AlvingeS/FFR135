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
    private:
        bool check_convergence();
        int classify();
        vector<vector<double>> weights;
        vector<vector<Neuron>> neurons;
        size_t nr_neurons;
};