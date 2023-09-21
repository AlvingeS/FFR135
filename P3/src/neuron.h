#pragma once
#include<vector>

typedef std::vector<double>* weights_vector_ptr;
typedef double* bias_ptr;

class Neuron {
    public:
        Neuron(double state, weights_vector_ptr weights, bias_ptr bias);
        void update_state(const std::vector<double> &input_signals);

        double get_state() {
            return this->state;
        };  

    private:
        double state;
        weights_vector_ptr weights;
        bias_ptr bias;
};