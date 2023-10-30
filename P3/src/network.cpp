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
    layer_heights[0] = arch.num_inputs;
    for (size_t i = 0; i < arch.num_hls(); i++) {
        layer_heights[i + 1] = arch.hl_sizes[i];
    }
    layer_heights[arch.num_hls()] = arch.num_outputs;
    num_layers = arch.num_hls() + 2;

    // Initialize pattern counts
    num_patterns = training_data.inputs.size();
    num_validation_patterns = validation_data.inputs.size();

    // Initialize weights and biases
    weights = double_tensor(num_layers, double_matrix(0, double_vector(0, 0.0)));
    biases = double_matrix(num_layers, double_vector(0, 0.0));
    
    // Initialize cumulative errors and products
    cumulative_products = double_tensor(num_layers, double_matrix(0, double_vector(0, 0.0)));
    cumulative_errors = double_matrix(num_layers, double_vector(0, 0.0));

    // Initialize velocities
    velocities_w = double_tensor(num_layers, double_matrix(0, double_vector(0, 0.0)));
    velocities_b = double_matrix(num_layers, double_vector(0, 0.0));

    // Initialize old velocities
    velocities_w_old = double_tensor(num_layers, double_matrix(0, double_vector(0, 0.0)));
    velocities_b_old = double_matrix(num_layers, double_vector(0, 0.0));

    // Initialize random number generators for weights
    std::default_random_engine generator;
    std::vector<std::normal_distribution<double>> distributions;

    for (size_t i = 1; i < num_layers; i++) {
        double std_dev = std::sqrt(1 / static_cast<double>(layer_heights[i - 1]));
        distributions.emplace_back(std::normal_distribution<double>(0.0, std_dev));
    }

    // Initialize hidden layer neurons
    neurons.reserve(num_layers);
    for (size_t l = 0; l < num_layers - 1; l++) {        
        weights[l] = double_matrix(this->layer_heights[l + 1], double_vector(this->layer_heights[l], 0.0));
        cumulative_products[l] = double_matrix(this->layer_heights[l + 1], double_vector(this->layer_heights[l], 0.0));
        velocities_w[l] = double_matrix(this->layer_heights[l + 1], double_vector(this->layer_heights[l], 0.0));
        velocities_w_old[l] = double_matrix(this->layer_heights[l + 1], double_vector(this->layer_heights[l], 0.0));

        biases[l] = double_vector(layer_heights[l], 0.0);
        cumulative_errors[l] = double_vector(layer_heights[l], 0.0);
        velocities_b[l] = double_vector(layer_heights[l], 0.0);
        velocities_b_old[l] = double_vector(layer_heights[l], 0.0);

        for (size_t i = 0; i < layer_heights[l + 1]; i++) {
            for (size_t j = 0; j < layer_heights[l]; j++) {
                weights[l][i][j] = distributions[l](generator);
            }
            neurons[l].emplace_back(Neuron(&weights[l][i], &biases[l][i]));
        }
    }
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
    for (size_t l = 0; l < this->num_layers - 1; l++) {
        double_vector layer_states(this->layer_heights[l + 1]);
        for (size_t i = 0; i < this->layer_heights[l + 1]; i++) {
            this->neurons[l][i].calculate_net_input(prev_layer_states);
            this->neurons[l][i].update_state();
            layer_states[i] = this->neurons[l][i].get_state();
        }
        prev_layer_states = layer_states;
    }
}

void Network::compute_errors(int target_index) {
    // Compute errors for output layer
    double_vector prev_error(this->arch.num_outputs, 0.0);
    size_t last_layer_index = this->num_layers - 2;

    for (size_t i = 0; i < this->arch.num_outputs; i++) {
        prev_error[i] = g_prime(this->neurons[last_layer_index][i].get_net_input()) * (this->training_data.targets[target_index][i] - this->neurons[last_layer_index][i].get_state());
        this->cumulative_errors[last_layer_index][i] += prev_error[i];
        
        // Accumulate products
        for (size_t n = 0; n < this->layer_heights[last_layer_index - 1]; n++) {
            this->cumulative_products[last_layer_index][i][n] += prev_error[i] * this->neurons[last_layer_index - 1][n].get_state();
        }
    }

    // Propagate errors backwards through hidden layers until and not including last hidden layer
    for (size_t l = last_layer_index - 1; l >= 1; l--) {      
        double_vector layer_errors(this->layer_heights[l], 0.0);

        for (size_t j = 0; j < this->layer_heights[l]; j++) {
            for (size_t i = 0; i < this->layer_heights[l + 1]; i++) {
                layer_errors[j] += prev_error[i] * this->weights[l][i][j] * g_prime(this->neurons[l][j].get_net_input());
            }
    
            this->cumulative_errors[l][j] += layer_errors[j];

            for (size_t n = 0; n < this->layer_heights[l - 1]; n++) {
                this->cumulative_products[l][j][n] += layer_errors[j] * this->neurons[l - 1][n].get_state();
            }
        }
        prev_error = layer_errors;
    }

    // Compute errors for first hidden layer
    for (size_t j = 0; j < this->layer_heights[0]; j++) {
        double_vector layer_errors = double_vector(this->layer_heights[0], 0.0); 
        for (size_t i = 0; i < this->layer_heights[1]; i++) {
            layer_errors[j] += prev_error[i] * this->weights[0][i][j] * g_prime(this->neurons[0][j].get_net_input());
        }

        this->cumulative_errors[0][j] += layer_errors[j];

        // Accumulate products
        for (size_t n = 0; n < this->arch.num_inputs; n++) {
            this->cumulative_products[0][j][n] += layer_errors[j] * this->training_data.inputs[target_index][n];
        }
    }
}

void Network::update_velocities(double learning_rate, int target_index, size_t batch_size) {
    this->velocities_w_old = this->velocities_w;
    this->velocities_b_old = this->velocities_b;

    double scaled_learning_rate = learning_rate / static_cast<double>(batch_size);

    for (size_t l = 0; l < this->num_layers - 1; l++) {
        for (size_t m = 0; m < this->layer_heights[l + 1]; m++) {
            for (size_t n = 0; n < this->layer_heights[l]; n++) {
                this->velocities_w[l][m][n] = scaled_learning_rate * this->cumulative_products[l][m][n];
                this->cumulative_products[l][m][n] = 0.0;
            }
            this->velocities_b[l][m] = scaled_learning_rate * this->cumulative_errors[l][m];
            this->cumulative_errors[l][m] = 0.0;
        }
    }    
}

void Network::update_weights_and_biases(double momentum) {
    // Update hidden layer weights and biases
    for (size_t l = 0; l < this->num_layers - 1; l++) {
        for (size_t m = 0; m < this->layer_heights[l + 1]; m++) {
            for (size_t n = 0; n < this->layer_heights[l]; n++) {
                this->weights[l][m][n] += this->velocities_w[l][m][n] + momentum * this->velocities_w_old[l][m][n];
            }
            this->biases[l][m] -= this->velocities_b[l][m];
        }
    }
}


void Network::validate(size_t epoch, bool measure_H, bool verbose) {
    this->C = 0.0;
    this->H = 0.0;

    if (measure_H) {
        for (size_t i = 0; i < this->num_patterns; i++) {
            this->propagate_forward(this->training_data.inputs[i]);
            for (size_t j = 0; j < this->arch.num_outputs; j++) {
                H += std::pow(this->training_data.targets[i][j] - this->get_output()[j], 2);
            }
        }

        H /= 2.0;
    }

    if (verbose) {
        if (measure_H) {
            std::cout << "Epoch: " << epoch << "  C: " << C << "  H: " << H << std::endl;
        } else {
            std::cout << "Epoch: " << epoch << "  C: " << C << std::endl;
        }
    }
}