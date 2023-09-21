#pragma once

#include "neuron.h"
#include <cstddef>
#include <vector>
#include <cstdint>
#include <cmath>

typedef std::vector<double> weights_vector;
typedef std::vector<weights_vector> weights_matrix;
typedef std::vector<double> bias_vector;
typedef std::vector<Neuron> neuron_vector;
typedef std::vector<int> target_vector;
typedef std::vector<double> error_vector;
typedef std::vector<std::vector<double>> input_matrix;

class Network {
    public:
        Network(size_t num_hl_neurons, size_t num_patterns, size_t num_validation_patterns);
        double get_output() {
            return this->ol_neuron.get_state();
        }

        void set_input_patterns(const input_matrix &input_patterns) {
            this->input_patterns = input_patterns;
        }

        void set_targets(const target_vector &targets) {
            this->targets = targets;
        }

        void set_validation_input_patterns(const input_matrix &input_patterns) {
            this->validation_input_patterns = input_patterns;
        }

        void set_validation_targets(const target_vector &targets) {
            this->validation_targets = targets;
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
        
        target_vector targets;
        input_matrix input_patterns;
        target_vector validation_targets;
        input_matrix validation_input_patterns;
        
        weights_matrix hl_weights;
        weights_vector ol_weights;
        bias_vector hl_biases;
        double ol_bias = 0.0;
        
        Neuron ol_neuron;
        neuron_vector hl_neurons;

        error_vector hl_errors;
        double ol_error = 0.0;
};