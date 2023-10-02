#pragma once

#include "neuron.h"
#include "utils.h"
#include <cstddef>
#include <vector>
#include <cstdint>
#include <cmath>

typedef std::vector<Neuron> neuron_vector;

struct weights_struct {
    double_matrix hl;
    double_vector ol;
};

struct biases_struct {
    double_vector hl;
    double ol;
};

struct velocities_struct {
    double_matrix hl;
    double_vector ol;
    double_vector hl_bias;
    double ol_bias;
};

struct neurons_struct {
    neuron_vector hl;
    neuron_vector ol;
};

struct errors_struct {
    double_vector hl;
    double ol;
};

class Network {
    public:
        Network(size_t num_hl_neurons, Data training_data, Data validation_data);
        
        double get_output() {
            return this->neurons.ol[0].get_state();
        }
        
        void train(double learning_rate, double min_learning_rate, double decay_rate, double momentum, size_t batch_size, size_t num_epoch, bool measure_H, bool verbose);

        std::vector<std::vector<double>> get_weights_hl() {
            return this->weights.hl;
        }

        std::vector<double> get_weights_ol() {
            return this->weights.ol;
        }

        std::vector<double> get_biases_hl() {
            return this->biases.hl;
        }

        double get_biases_ol() {
            return this->biases.ol;
        }

        void export_validation_results();

    private:
        
        double g_prime(double x) {
            return 1 - std::pow(std::tanh(x), 2);
        }

        void propagate_forward(const std::vector<double> &input_signals);
        std::vector<double> get_hl_states();
        void compute_errors(int target_index);
        void update_velocities(double learning_rate, int target_index, size_t batch_size);
        void update_weights_and_biases(double momentum);
        void validate(size_t epoch, bool measure_H, bool verbose);

        size_t num_inputs = 2;
        size_t num_hl_neurons;
        size_t num_patterns;
        size_t num_validation_patterns;
        
        // Data storage
        Data training_data;
        Data validation_data;
        
        weights_struct weights;
        biases_struct biases;

        velocities_struct velocities;
        velocities_struct old_velocities;
        
        neurons_struct neurons;

        errors_struct errors;

        double C = 0.0;
        double H = 0.0;
};