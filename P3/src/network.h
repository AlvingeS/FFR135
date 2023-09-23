#pragma once

#include "neuron.h"
#include "utils.h"
#include <cstddef>
#include <vector>
#include <cstdint>
#include <cmath>

typedef std::vector<Neuron> neuron_vector;

class Network {
    public:
        Network(size_t num_hl_neurons, Data training_data, Data validation_data);
        
        double get_output() {
            return this->ol_neuron.get_state();
        }
        
        void train(double learning_rate, size_t batch_size, size_t num_epoch);

    private:
        std::vector<double> get_hl_states();
        
        double g_prime(double x) {
            return 1 - std::pow(std::tanh(x), 2);
        }

        void propagate_forward(const std::vector<double> &input_signals);
        void propagate_backward(int target_index);
        void compute_output_error(int target_index);
        void compute_hidden_layer_errors();
        void update_weights_and_biases(double learning_rate, size_t batch_size);

        void validate();

        size_t num_inputs = 2;
        size_t num_hl_neurons;
        size_t num_patterns;
        size_t num_validation_patterns;
        
        // Data storage
        Data training_data;
        Data validation_data;
        
        double_matrix hl_weights;
        double_vector ol_weights;
        double_vector hl_biases;
        double ol_bias = 0.0;
        
        Neuron ol_neuron;
        neuron_vector hl_neurons;

        double_vector hl_errors;
        double ol_error = 0.0;

        double C = 0.0;
        double H = 0.0;
};