#include<vector>

#pragma once

using namespace std;

class Neuron {
    public:
        Neuron(int state, vector<double>& neuron_weights);
        void set_neuron_weights(vector<double>& neuron_weights);
        void update_state(const vector<double> input_signals);
        void set_state(int state);
        int get_state();
    private:
        int state;
        vector<double>& neuron_weights;
};