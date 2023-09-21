#pragma once

#include "neuron.h"
#include <cstddef>
#include <vector>
#include <cstdint>

typedef std::vector<double> weights_vector;
typedef std::vector<weights_vector> weights_matrix;
typedef std::vector<double> bias_vector;
typedef std::vector<Neuron> neuron_vector;

class Network {
    public:
        Network(size_t num_hl_neurons);
        void forward_call(const std::vector<double> &input_signals);
        double get_output() {
            return this->ol_neuron.get_state();
        }

    private:
        std::vector<double> get_hl_states();

        size_t num_inputs = 2;
        size_t num_hl_neurons;
        
        weights_matrix hl_weights;
        bias_vector hl_biases;
        weights_vector ol_weights;
        double ol_bias;
        
        Neuron ol_neuron;
        neuron_vector hl_neurons;
};