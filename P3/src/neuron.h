#pragma once
#include<vector>

typedef std::vector<double>* weights_vector_ptr;
typedef double* bias_ptr;

class Neuron {
    public:
        Neuron(weights_vector_ptr weights, bias_ptr bias);
        
        void calculate_net_input(const std::vector<double> &input_signals);
        void update_state();

        double get_state() {
            return this->state;
        };

        double get_net_input() {
            return this->net_input;
        };

    private:


        double net_input = 0.0;
        double state = 0.0;
        weights_vector_ptr weights;
        bias_ptr bias;
};