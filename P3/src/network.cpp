#include "network.h"
#include "matrix.h"
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

// Constructor for network
Network::Network(arch_struct arch, Data training_data, Data validation_data)
    : arch(arch),
      layer_heights(arch.num_hls() + 2, 0),
      L(arch.num_hls() + 2 - offset),
      training_data(training_data),
      validation_data(validation_data),   
      weights(L),
      cumulative_products(L),
      velocities_w(L),
      velocities_w_old(L),
      neuron_states(L),
      net_inputs(L),
      biases(L),
      cumulative_errors(L),
      velocities_b(L),
      velocities_b_old(L) {
 
    // Initialize layer heights
    this->layer_heights[0] = arch.num_inputs;
    for (size_t i = 0; i < arch.num_hls(); i++) {
        this->layer_heights[i + 1] = arch.hl_sizes[i];
    }
    this->layer_heights[arch.num_hls() + 1] = arch.num_outputs;

    // Initialize pattern counts
    num_patterns = training_data.inputs.getRows();
    num_validation_patterns = validation_data.inputs.getRows();

    // Generator
    for (size_t l = 0; l < L; l++) {
        weights[l] = Matrix<double>(this->layer_heights[l + 1], this->layer_heights[l], 0.0);
        cumulative_products[l] = Matrix<double>(this->layer_heights[l + 1], this->layer_heights[l], 0.0);
        velocities_w[l] = Matrix<double>(this->layer_heights[l + 1], this->layer_heights[l], 0.0);
        velocities_w_old[l] = Matrix<double>(this->layer_heights[l + 1], this->layer_heights[l], 0.0);

        net_inputs[l] = Vector<double>(this->layer_heights[l + 1], 0.0);
        neuron_states[l] = Vector<double>(this->layer_heights[l + 1], 0.0);
        biases[l] = Vector<double>(layer_heights[l + 1], 0.0);
        cumulative_errors[l] = Vector<double>(layer_heights[l + 1], 0.0);
        velocities_b[l] = Vector<double>(layer_heights[l + 1], 0.0);
        velocities_b_old[l] = Vector<double>(layer_heights[l + 1], 0.0);

        std::default_random_engine generator;
        double std_dev = std::sqrt(1 / static_cast<double>(layer_heights[l]));
        std::normal_distribution<double> distribution = std::normal_distribution<double>(0.0, std_dev);

        for (size_t i = 0; i < layer_heights[l + 1]; i++) {
            for (size_t j = 0; j < layer_heights[l]; j++) {
                weights[l][i][j] = distribution(generator);
            }
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

        if (this->H < H_min) {
            H_min = this->H;
        }
    }
}

void Network::propagate_forward(const Vector<double> &input_signals) {
    for (size_t l = 1; l <= L; l++) {
        Vector<double> signals(this->layer_heights[l - 1], 0.0);
        signals = (l - 1 == 0) ? input_signals : neuron_states[l - 1 - offset];

        net_inputs[l - offset] = (weights[l - offset] * signals - biases[l - offset]);
        neuron_states[l - offset] = net_inputs[l - offset].apply_function(g);
    }
}

void Network::compute_errors(int target_index) {
    // Compute errors for output layer
    Vector<double> prev_errors(this->layer_heights[L], 0.0);
    prev_errors = (net_inputs[L - offset].apply_function(g_prime)) % (this->training_data.targets[target_index] - this->neuron_states[L - offset]);
    cumulative_errors[L - offset] += prev_errors;
    cumulative_products[L - offset] += Matrix<double>::outer_product(prev_errors, neuron_states[L - 1 - offset]);

    for (size_t l = L; l >= 2; l--) {
        Vector<double> layer_errors(this->layer_heights[l - 1], 0.0);

        layer_errors = net_inputs[l - 1 - offset].apply_function(g_prime) % (weights[l - offset].transpose() * prev_errors);
        cumulative_errors[l - 1 - offset] += layer_errors;
        Vector<double> states = (l - 2 == 0) ? this->training_data.inputs[target_index] : neuron_states[l - 2 - offset];
        cumulative_products[l - 1 - offset] += Matrix<double>::outer_product(layer_errors, states);

        prev_errors = layer_errors;
    }
}

void Network::update_velocities(double learning_rate, int target_index, size_t batch_size) {
    this->velocities_w_old = this->velocities_w;
    this->velocities_b_old = this->velocities_b;
        
    double scaled_learning_rate = learning_rate / static_cast<double>(batch_size);

    for (size_t l = 1; l <= L; l++) {
        velocities_w[l - offset] = this->cumulative_products[l - offset] * scaled_learning_rate;
        velocities_b[l - offset] = this->cumulative_errors[l - offset] * scaled_learning_rate;
    }
}

void Network::update_weights_and_biases(double momentum) {
    for (size_t l = 1; l <= L; l++) {
        weights[l - offset] += velocities_w[l - offset] + velocities_w_old[l - offset] * momentum;
        biases[l - offset] -= velocities_b[l - offset];
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