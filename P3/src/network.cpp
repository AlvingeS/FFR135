#include "network.h"
#include "neuron.h"
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>

// Constructor for network
Network::Network(arch_struct arch, Data training_data, Data validation_data)
    : arch(arch),
      training_data(training_data),
      validation_data(validation_data) {

    // Initialize layer heights
    layer_heights = int_vector(arch.num_hls() + 1, 0);
    for (size_t i = 0; i < arch.num_hls(); i++) {
        layer_heights[i] = arch.hl_sizes[i];
    }
    layer_heights[arch.num_hls()] = arch.num_outputs;
    num_layers = arch.num_hls() + 1;

    // Initialize pattern counts
    num_patterns = training_data.inputs.size();
    num_validation_patterns = validation_data.inputs.size();

    // Initialize weights and biases
    weights.hl = double_matrix(arch.hl_sizes[0], double_vector(arch.num_inputs, 0.0));
    weights.ol = double_matrix(arch.num_outputs, double_vector(arch.hl_sizes[0], 0.0));
    biases.hl = double_vector(arch.hl_sizes[0], 0.0);
    biases.ol = double_vector(arch.num_outputs, 0.0);

    // Initialize velocities
    velocities.hl = double_matrix(arch.hl_sizes[0], double_vector(arch.num_inputs, 0.0));
    velocities.ol = double_matrix(arch.num_outputs, double_vector(arch.hl_sizes[0], 0.0));
    velocities.hl_bias = double_vector(arch.hl_sizes[0], 0.0);
    velocities.ol_bias = double_vector(arch.num_outputs, 0.0);

    // Initialize old velocities
    old_velocities.hl = double_matrix(arch.hl_sizes[0], double_vector(arch.num_inputs, 0.0));
    old_velocities.ol = double_matrix(arch.num_outputs, double_vector(arch.hl_sizes[0], 0.0));

    // Initialize random number generators for weights
    double mean = 0.0;
    double std_dev_hl = 1.0 / sqrt(static_cast<double>(arch.num_inputs));
    double std_dev_ol = 1.0 / sqrt(static_cast<double>(arch.hl_sizes[0]));
    std::default_random_engine generator;
    std::normal_distribution<double> distribution_hl(mean, std_dev_hl);
    std::normal_distribution<double> distribution_ol(mean, std_dev_ol);

    // Initialize hidden layer neurons
    neurons.hl.reserve(arch.hl_sizes[0]);
    neurons.ol.reserve(1);
    for (size_t i = 0; i < arch.hl_sizes[0]; i++) {
        for (size_t j = 0; j < arch.num_inputs; j++) {
            weights.hl[i][j] = distribution_hl(generator);
        }
        neurons.hl.emplace_back(&weights.hl[i], &biases.hl[i]);
    }

    // Initialize output layer neurons
    for (size_t i = 0; i < arch.num_outputs; i++) {
        for (size_t j = 0; j < arch.hl_sizes[0]; j++) {
            weights.ol[i][j] = distribution_ol(generator);
        }
        neurons.ol.emplace_back(&weights.ol[i], &biases.ol[i]);
    }

    // Initialize cumulative errors and products
    cumulative_errors.hl = double_vector(arch.hl_sizes[0], 0.0);
    cumulative_errors.ol = double_vector(arch.num_outputs, 0.0);
    cumulative_products.hl = double_matrix(arch.hl_sizes[0], double_vector(arch.num_inputs, 0.0));
    cumulative_products.ol = double_matrix(arch.num_outputs, double_vector(arch.hl_sizes[0], 0.0));
}

void Network::train(double learning_rate, double momentum, size_t batch_size, size_t num_epochs, bool measure_H, bool verbose) {
    double H_min = 3000;

    for (size_t i = 0; i < num_epochs; i++) {
        for (size_t j = 0; j < this-> num_patterns; j++) {

            this->propagate_forward(this->training_data.inputs[j]);
            this->compute_errors(j);

            if ((j + 1) % batch_size == 0) {
                this->update_velocities(learning_rate, j, batch_size);
                this->update_weights_and_biases(momentum);
            }
        }

        this->validate(i, measure_H, verbose);

        if (this->H < 2100) {
            break;
        }

        if (this->H < H_min) {
            H_min = this->H;
        }
    }
}

void Network::propagate_forward(const double_vector &input_signals) {
    double_vector prev_layer_states = input_signals;
    for (size_t l = 0; l < this->num_layers; l++) {
        double_vector layer_states(this->layer_heights[l]);
        for (size_t i = 0; i < this->layer_heights[l]; i++) {
            this->neurons[l][i].calculate_net_input(prev_layer_states);
            this->neurons[l][i].update_state();
            layer_states[i] = this->neurons[l][i].get_state();
        }
        prev_layer_states = layer_states;
    }
};

void Network::compute_errors(int target_index) {

    double_vector errors_ol(this->arch.num_outputs, 0.0);
    for (size_t m = 0; m < this->arch.num_outputs; m++) {
        errors_ol[m] = g_prime(this->neurons.[l][m].get_net_input()) * (this->training_data.targets[target_index][m] - this->neurons.ol[m].get_state());
        this->cumulative_errors.ol[m] += errors_ol[m];
        for (size_t n = 0; n < this->arch.hl_sizes[0]; n++) {
            this->cumulative_products.ol[m][n] += errors_ol[m] * this->neurons.hl[n].get_state();
        }
    }

    // Compute hidden layer error and accumulate
    double_vector errors_hl(this->arch.hl_sizes[0], 0.0); 
    for (size_t j = 0; j < this->arch.hl_sizes[0]; j++) {
        for (size_t i = 0; i < this->arch.num_outputs; i++) {
            errors_hl[j] += errors_ol[i] * this->weights.ol[i][j] * g_prime(this->neurons.hl[j].get_net_input());
        }
        this->cumulative_errors.hl[j] += errors_hl[j];
    }

    // Accumulate hidden layer products
    for (size_t m = 0; m < this->arch.hl_sizes[0]; m++) {
        for (size_t n = 0; n < this->arch.num_inputs; n++) {
            this->cumulative_products.hl[m][n] += errors_hl[m] * this->training_data.inputs[target_index][n];
        }
    }
}

void Network::compute_errors(int target_index) {

    // Compute errors for output layer
    double_vector prev_error(this->arch.num_outputs, 0.0);
    size_t last_layer_index = this->num_layers - 1;

    for (size_t i = 0; i < this->arch.num_outputs; i++) {
        prev_error[i] = g_prime(this->neurons[last_layer_index][i].get_net_input()) * (this->training_data.targets[target_index][i] - this->neurons[last_layer_index][i].get_state());
        this->cumulative_errors[last_layer_index][i] += prev_error[i];
        
        // Accumulate products
        for (size_t n = 0; n < this->layer_heights[last_layer_index - 1]; n++) {
            this->cumulative_products[last_layer_index][i][n] += prev_error[i] * this->neurons[last_layer_index - 1][n].get_state();
        }
    }

    // Propagate errors backwards
    for (size_t l = last_layer_index; l >= 1; l--) {      
        double_vector layer_errors(this->layer_heights[l], 0.0);

        for (size_t j = 0; j < this->layer_heights[l - 1]; j++) {
            for (size_t i = 0; i < this->layer_heights[l]; i++) {
                layer_errors[j] += prev_error[i] * this->weights[l][i][j] * g_prime(this->neurons[l - 1][j].get_net_input());
            }
    
            this->cumulative_errors[l][j] += layer_errors[j];

            for (size_t i = 0; n < this->; n++) {
                this->cumulative_products[l][i][j] += layer_errors[m] * this->neurons.hl[n].get_state();
            }
        }
    }
}



void Network::update_velocities(double learning_rate, int target_index, size_t batch_size) {
    this->old_velocities.hl = this->velocities.hl;
    this->old_velocities.ol = this->velocities.ol;

    double scaled_learning_rate = learning_rate / static_cast<double>(batch_size);

    // Update hidden layer velocities
    for (size_t m = 0; m < this->arch.hl_sizes[0]; m++) {
        for (size_t n = 0; n < this->arch.num_inputs; n++) {
            this->velocities.hl[m][n] = scaled_learning_rate * this->cumulative_products.hl[m][n];
        }
        this->velocities.hl_bias[m] = scaled_learning_rate * this->cumulative_errors.hl[m];
    }

    // Update output layer velocities
    for (size_t m = 0; m < this->arch.num_outputs; m ++) {
        for (size_t n = 0; n < this->arch.hl_sizes[0]; n++) {
            this->velocities.ol[m][n] = scaled_learning_rate * this->cumulative_products.ol[m][n];
        }
        this->velocities.ol_bias[m] = scaled_learning_rate * this->cumulative_errors.ol[m];
    }

    // Reset cumulative errors and products
    this->cumulative_errors.hl = double_vector(this->arch.hl_sizes[0], 0.0);
    this->cumulative_errors.ol = double_vector(this->arch.num_outputs, 0.0);
    this->cumulative_products.hl = double_matrix(this->arch.hl_sizes[0], double_vector(this->arch.num_inputs, 0.0));
    this->cumulative_products.ol = double_matrix(this->arch.num_outputs, double_vector(this->arch.hl_sizes[0], 0.0));
}

void Network::update_weights_and_biases(double momentum) {
    // Update hidden layer weights and biases
    for (size_t m = 0; m < this->arch.hl_sizes[0]; m++) {
        for (size_t n = 0; n < this->arch.num_inputs; n++) {
            this->weights.hl[m][n] += this->velocities.hl[m][n] + momentum * this->old_velocities.hl[m][n];
        }
        this->biases.hl[m] -= this->velocities.hl_bias[m];
    }

    // Update output layer weights and biases
    for (size_t m = 0; m < this->arch.num_outputs; m++) {
        for (size_t n = 0; n < this->arch.hl_sizes[0]; n++) {
            this->weights.ol[m][n] += this->velocities.ol[m][n] + momentum * this->old_velocities.ol[m][n];
        }
        this->biases.ol[m] -= this->velocities.ol_bias[m];
    }
}


void Network::validate(size_t epoch, bool measure_H, bool verbose) {
    this->C = 0.0;
    this->H = 0.0;

    if (measure_H) {
        for (size_t i = 0; i < this->num_patterns; i++) {
            this->propagate_forward(this->training_data.inputs[i]);
            H += std::pow(this->training_data.targets[i][0] - this->get_output(), 2);
        }

        H /= 2.0;
    }
    
    for (size_t i = 0; i < this->num_validation_patterns; i++) {
        this->propagate_forward(this->validation_data.inputs[i]);

        int classification = (this->get_output() > 0) ? 1 : -1;

        C += std::abs(classification - this->validation_data.targets[i][0]);
    }
    
    C /= 2 * static_cast<double>(this->num_validation_patterns);

    if (verbose) {
        if (measure_H) {
            std::cout << "Epoch: " << epoch << "  C: " << C << "  H: " << H << std::endl;
        } else {
            std::cout << "Epoch: " << epoch << "  C: " << C << std::endl;
        }
    }
}