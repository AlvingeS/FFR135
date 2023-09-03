#pragma once
#include<vector>

typedef std::vector<double>* vector_double_ptr;

class Neuron {
    public:
        Neuron(int state, vector_double_ptr neuron_weights_ptr, double bias);
        void update_state(std::vector<int> input_signals);
        void set_state(int state);
        int get_state();
        void set_bias(double bias);
        double get_bias();
    private:
        int state;
        vector_double_ptr neuron_weights_ptr;
        double bias;
};