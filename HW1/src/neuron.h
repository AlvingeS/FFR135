#pragma once
#include<vector>

typedef std::vector<double>& vector_double_ref;

class Neuron {
    public:
        Neuron(int state, vector_double_ref neuron_weights);
        void set_neuron_weights(vector_double_ref neuron_weights);
        void update_state(std::vector<double> input_signals);
        void set_state(int state);
        int get_state();
    private:
        int state;
        vector_double_ref neuron_weights;
};