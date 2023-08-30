#include<vector>

#pragma once

using namespace std;

class Neuron {
    public:
        Neuron(vector<double>& neuron_weights, double threshold);
        void update_state(const vector<double> input_signals);
    private:
        vector<double>& neuron_weights;
        double threshold;
        int state;
};