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
      weights_transposed(L),
      cumulative_products(L),
      velocities_w(L),
      velocities_w_old(L),
      deltas(L),
      neuron_states(L),
      net_inputs(L),
      biases(L),
      cumulative_errors(L),
      velocities_b(L),
      output_diff(arch.num_outputs, 0.0),
      output_element_wise_diff(arch.num_outputs, 0.0),
      internal_element_wise_diff(L) {
 
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
        weights_transposed[l] = Matrix<double>(this->layer_heights[l], this->layer_heights[l + 1], 0.0);
        cumulative_products[l] = Matrix<double>(this->layer_heights[l + 1], this->layer_heights[l], 0.0);
        velocities_w[l] = Matrix<double>(this->layer_heights[l + 1], this->layer_heights[l], 0.0);
        velocities_w_old[l] = Matrix<double>(this->layer_heights[l + 1], this->layer_heights[l], 0.0);

        deltas[l] = Vector<double>(this->layer_heights[l + 1], 0.0);
        net_inputs[l] = Vector<double>(this->layer_heights[l + 1], 0.0);
        internal_element_wise_diff[l] = Vector<double>(this->layer_heights[l + 1], 0.0);
        neuron_states[l] = Vector<double>(this->layer_heights[l + 1], 0.0);
        biases[l] = Vector<double>(layer_heights[l + 1], 0.0);
        cumulative_errors[l] = Vector<double>(layer_heights[l + 1], 0.0);
        velocities_b[l] = Vector<double>(layer_heights[l + 1], 0.0);

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

void Network::train(double learning_rate, double momentum, size_t batch_size, size_t num_epochs, bool measure_H, bool verbose, double lookup_tol) {
    double H_min = 3000;
    this->scaled_learning_rate = learning_rate / static_cast<double>(batch_size);

    for (size_t i = 0; i < num_epochs; i++) {
        for (size_t j = 0; j < this-> num_patterns; j++) {

            this->propagate_forward(this->training_data.inputs[j], lookup_tol);
            this->compute_errors(j, lookup_tol);

            if ((j + 1) % batch_size == 0) {
                this->update_velocities(learning_rate, j, batch_size);
                this->update_weights_and_biases(momentum);
            }
        }

        this->validate(i, measure_H, verbose, lookup_tol);

        if (this->H < H_min) {
            H_min = this->H;
        }
    }
}

void Network::propagate_forward(const Vector<double> &input_signals, double lookup_tol) {
    for (size_t l = 1; l <= L; l++) {
        if (l - 1 == 0) {
            net_inputs[l - offset] = weights[l - offset] * input_signals;
            net_inputs[l- offset] -= biases[l - offset];
        } else {
            net_inputs[l - offset] = weights[l - offset] * neuron_states[l - 1 - offset];
            net_inputs[l - offset] -= biases[l - offset];
        }

        neuron_states[l - offset] = net_inputs[l - offset].apply_function(g);
    }
}

void Network::compute_errors(int target_index, double lookup_tol) {
    // Compute errors for output layer
    output_diff = this->training_data.targets[target_index];
    output_diff -= this->neuron_states[L - offset];
    deltas[L - offset] = (net_inputs[L - offset].apply_function(g_prime)).in_place_elementwise_multiplication(output_diff, output_element_wise_diff);
    
    cumulative_errors[L - offset] += deltas[L - offset];
    Matrix<double>::outer_product(deltas[L - offset], neuron_states[L - 1 - offset], cumulative_products[L - offset]);

    for (size_t l = L; l >= 2; l--) {
        deltas[l - 1 - offset] = (net_inputs[l - 1 - offset].apply_function(g_prime)).in_place_elementwise_multiplication(weights[l - offset].transpose(weights_transposed[l - offset]) * deltas[l - offset], internal_element_wise_diff[l - 1 - offset]);
        cumulative_errors[l - 1 - offset] += deltas[l - 1 - offset];

        if (l - 2 == 0) {
            Matrix<double>::outer_product(deltas[l - 1 - offset], this->training_data.inputs[target_index], cumulative_products[l - 1 - offset]);
        } else {
            Matrix<double>::outer_product(deltas[l - 1 - offset], neuron_states[l - 2 - offset], cumulative_products[l - 1 - offset]);
        }
    }
}

void Network::update_velocities(double learning_rate, int target_index, size_t batch_size) {
    this->velocities_w_old = this->velocities_w;
        
    for (size_t l = 1; l <= L; l++) {
        velocities_w[l - offset] = this->cumulative_products[l - offset] *= scaled_learning_rate;
        velocities_b[l - offset] = this->cumulative_errors[l - offset] *= scaled_learning_rate;
    
        cumulative_errors[l - offset].reset();
        cumulative_products[l - offset].reset();
    }
}

void Network::update_weights_and_biases(double momentum) {
    for (size_t l = 1; l <= L; l++) {
        weights[l - offset] += velocities_w[l - offset];
        weights[l - offset] += velocities_w_old[l - offset] *= momentum;
        biases[l - offset] -= velocities_b[l - offset];
    }
}

void Network::validate(size_t epoch, bool measure_H, bool verbose, double lookup_tol) {
    this->C = 0.0;
    this->H = 0.0;

    if (measure_H) {
        for (size_t i = 0; i < this->num_patterns; i++) {
            this->propagate_forward(this->training_data.inputs[i], lookup_tol);
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