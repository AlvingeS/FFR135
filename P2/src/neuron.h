#pragma once
#include<vector>

typedef std::vector<double>* vector_double_ptr;

class Neuron {
    public:
        Neuron(int state, vector_double_ptr neuron_weights_ptr, double bias);
        void update_state(std::vector<int> &input_signals);
        
        void set_state(int state) {
            this->state = state;
        };

        int get_state() {
            return this->state;
        };

        void set_bias(double bias) {
            this->bias = bias;
        };

        double get_bias() {
            return this->bias;
        };        

    private:
        int state;
        vector_double_ptr neuron_weights_ptr;
        double bias;
};