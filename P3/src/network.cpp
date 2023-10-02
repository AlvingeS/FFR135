#include "network.h"
#include "neuron.h"
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>

// Constructor for network
Network::Network(size_t num_hl_neurons, Data training_data, Data validation_data)
    : num_hl_neurons(num_hl_neurons),
      training_data(training_data),
      validation_data(validation_data) {
    
    this->num_patterns = this->training_data.inputs.size();
    this->num_validation_patterns = this->validation_data.inputs.size();

    this->weights.hl = double_matrix(this->num_hl_neurons, double_vector(this->num_inputs, 0.0));
    this->weights.ol = double_vector(this->num_hl_neurons, 0.0);

    this->biases.hl = double_vector(this->num_hl_neurons, 0.0);
    this->biases.ol = 0.0;

    this->velocities.hl = double_matrix(this->num_hl_neurons, double_vector(this->num_inputs, 0.0));
    this->velocities.ol = double_vector(this->num_hl_neurons, 0.0);
    this->velocities.hl_bias = double_vector(this->num_hl_neurons, 0.0);
    this->velocities.ol_bias = 0.0;

    this->old_velocities.hl = double_matrix(this->num_hl_neurons, double_vector(this->num_inputs, 0.0));
    this->old_velocities.ol = double_vector(this->num_hl_neurons, 0.0);

    double mean = 0.0;
    double std_dev_hl = 1.0 / sqrt(static_cast<double>(this->num_inputs));
    double std_dev_ol = 1.0 / sqrt(static_cast<double>(this->num_hl_neurons));
    std::default_random_engine generator;
    std::normal_distribution<double> distribution_hl(mean, std_dev_hl);
    std::normal_distribution<double> distribution_ol(mean, std_dev_ol);

    this->neurons.hl.reserve(this->num_hl_neurons);
    this->neurons.ol.reserve(1);
    
    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        this->weights.hl[i][0] = distribution_hl(generator);
        this->weights.hl[i][1] = distribution_hl(generator);
        this->weights.ol[i] = distribution_ol(generator);

        this->neurons.hl.emplace_back(&this->weights.hl[i], &this->biases.hl[i]);
    }

    this->neurons.ol.emplace_back(&this->weights.ol, &this->biases.ol);

    this->errors.hl = double_vector(this->num_hl_neurons, 0.0);
    this->errors.ol = 0.0;

    this->deltas.hl = double_matrix(this->num_hl_neurons, double_vector(this->num_inputs, 0.0));
    this->deltas.ol = double_vector(this->num_hl_neurons, 0.0);
};

void Network::train(double learning_rate, double min_learning_rate, double decay_rate, double momentum, size_t batch_size, size_t num_epochs, bool measure_H, bool verbose) {
    double H_min = 3000;

    for (size_t i = 0; i < num_epochs; i++) {
        learning_rate = std::max(learning_rate * decay_rate, min_learning_rate);
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

        if (this->H < 2100) {
            std::cout << std::endl;
            std::cout << " --------------* H < 2100 *-------------- " << std::endl;
            std::cout << std::endl;
            break;
        }
    }

    std::cout << "H_min: " << H_min << std::endl;

}

void Network::propagate_forward(const double_vector &input_signals) {
    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        this->neurons.hl[i].calculate_net_input(input_signals);
        this->neurons.hl[i].update_state();
    }

    this->neurons.ol[0].calculate_net_input(this->get_hl_states());
    this->neurons.ol[0].update_state();
};

double_vector Network::get_hl_states() {
    double_vector hl_states(this->num_hl_neurons);

    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        hl_states[i] = this->neurons.hl[i].get_state();
    }

    return hl_states;
};

void Network::compute_errors(int target_index) {
    double error_ol = g_prime(this->neurons.ol[0].get_net_input()) * (this->training_data.targets[target_index] - this->neurons.ol[0].get_state());
    this->errors.ol += error_ol;

    for (size_t n = 0; n < this->num_hl_neurons; n++) {
        this->deltas.ol[n] += error_ol * this->neurons.hl[n].get_state();
    }

    double_vector errors_hl = double_vector(this->num_hl_neurons, 0.0); 

    for (size_t j = 0; j < this->num_hl_neurons; j++) {
        errors_hl[j] = this->errors.ol * this->weights.ol[j] * g_prime(this->neurons.hl[j].get_net_input());
        errors.hl[j] += errors_hl[j];
    }

    for (size_t m = 0; m < this->num_hl_neurons; m++) {
        for (size_t n = 0; n < this->num_inputs; n++) {
            this->deltas.hl[m][n] += errors_hl[m] * this->training_data.inputs[target_index][n];
        }
    }
};

void Network::update_velocities(double learning_rate, int target_index, size_t batch_size) {
    this->old_velocities.hl = this->velocities.hl;
    this->old_velocities.ol = this->velocities.ol;

    double batch_size_inverse = 1.0 / static_cast<double>(batch_size);
    double product = learning_rate * batch_size_inverse;

    for (size_t m = 0; m < this->num_hl_neurons; m++) {
        for (size_t n = 0; n < this->num_inputs; n++) {
            this->velocities.hl[m][n] = product * this->deltas.hl[m][n];
        }

        this->velocities.ol[m] = product * this->deltas.ol[m];
        
        this->velocities.hl_bias[m] = product * this->errors.hl[m];
    }

    this->velocities.ol_bias = product * this->errors.ol;

    this->errors.hl = double_vector(this->num_hl_neurons, 0.0);
    this->errors.ol = 0.0;
    this->deltas.hl = double_matrix(this->num_hl_neurons, double_vector(this->num_inputs, 0.0));
    this->deltas.ol = double_vector(this->num_hl_neurons, 0.0);
}

void Network::update_weights_and_biases(double momentum) {
    for (size_t m = 0; m < this->num_hl_neurons; m++) {

        for (size_t n = 0; n < this->num_inputs; n++) {
            this->weights.hl[m][n] += this->velocities.hl[m][n] + momentum * this->old_velocities.hl[m][n];
        }

        this->weights.ol[m] += this->velocities.ol[m] + momentum * this->old_velocities.ol[m];

        this->biases.hl[m] -= this->velocities.hl_bias[m];
        this->biases.ol -= this->velocities.ol_bias;
    }
}

void Network::validate(size_t epoch, bool measure_H, bool verbose) {
    this->C = 0.0;
    this->H = 0.0;
    int consecutive_minus_ones = 0;

    if (measure_H) {
        for (size_t i = 0; i < this->num_patterns; i++) {
            this->propagate_forward(this->training_data.inputs[i]);
            H += std::pow(this->training_data.targets[i] - this->get_output(), 2);
        }

        H /= 2.0;
    }
    
    for (size_t i = 0; i < this->num_validation_patterns; i++) {
        this->propagate_forward(this->validation_data.inputs[i]);

        int classification = (this->get_output() > 0) ? 1 : -1;

        if (classification == -1) {
            consecutive_minus_ones++;
        } else {
            consecutive_minus_ones = 0;
        }

        C += std::abs(classification - this->validation_data.targets[i]);

        // Stop the validation if the network classifies all patterns as -1
        if (consecutive_minus_ones == 1000) {
            break;
        }
    }

    if (consecutive_minus_ones == 1000) {
        C = 1.0;
    } else {
        C /= 2 * static_cast<double>(this->num_validation_patterns);
    }

    if (verbose) {
        if (measure_H) {
            std::cout << "Epoch: " << epoch << "  C: " << C << "  H: " << H << std::endl;
        } else {
            std::cout << "Epoch: " << epoch << "  C: " << C << std::endl;
        }
    }
}

void Network::export_validation_results() {
    std::ofstream file("output_parameters/validation_results.csv");
    file << "X,Y,Label" << std::endl;

    for (size_t i = 0; i < this->num_validation_patterns; i++) {
        this->propagate_forward(this->validation_data.inputs[i]);

        int classification = (this->get_output() > 0) ? 1 : -1;

        file << this->validation_data.inputs[i][0] << "," 
             << this->validation_data.inputs[i][1] << ","
             << classification << std::endl;
    }

    file.close();
}