#include "network.h"
#include "neuron.h"
#include <random>
#include <cmath>
#include <iostream>

// Constructor for network
Network::Network(size_t num_hl_neurons, Data training_data, Data validation_data)
    : num_hl_neurons(num_hl_neurons),
      training_data(training_data),
      validation_data(validation_data),
      hl_weights(num_hl_neurons, double_vector(2)),
      ol_weights(num_hl_neurons),
      hl_biases(num_hl_neurons, 0.0),
      hl_velocity(num_hl_neurons, double_vector(2, 0.0)),
      ol_velocity(num_hl_neurons, 0.0),
      hl_bias_velocity(num_hl_neurons, 0.0),
      old_hl_velocity(num_hl_neurons, double_vector(2, 0.0)),
      old_ol_velocity(num_hl_neurons, 0.0),
      old_hl_bias_velocity(num_hl_neurons, 0.0),
      ol_neuron(&this->ol_weights, &this->ol_bias),
      hl_errors(num_hl_neurons, 0.0) {
    
    this->num_patterns = this->training_data.inputs.size();
    this->num_validation_patterns = this->validation_data.inputs.size();

    double mean = 0.0;
    double std_dev_hl = 1.0 / sqrt(static_cast<double>(this->num_inputs));
    double std_dev_ol = 1.0 / sqrt(static_cast<double>(this->num_hl_neurons));
    std::default_random_engine generator;
    std::normal_distribution<double> distribution_hl(mean, std_dev_hl);
    std::normal_distribution<double> distribution_ol(mean, std_dev_ol);

    this->hl_neurons.reserve(this->num_hl_neurons);
    
    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        this->hl_weights[i][0] = distribution_hl(generator);
        this->hl_weights[i][1] = distribution_hl(generator);
        this->ol_weights[i] = distribution_ol(generator);

        this->hl_neurons.emplace_back(&this->hl_weights[i], &this->hl_biases[i]);
    }
};

void Network::propagate_forward(const double_vector &input_signals) {
    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        this->hl_neurons[i].calculate_net_input(input_signals);
        this->hl_neurons[i].update_state();
    }

    this->ol_neuron.calculate_net_input(this->get_hl_states());
    this->ol_neuron.update_state();
};

double_vector Network::get_hl_states() {
    double_vector hl_states(this->num_hl_neurons);

    for (size_t i = 0; i < this->num_hl_neurons; i++) {
        hl_states[i] = this->hl_neurons[i].get_state();
    }

    return hl_states;
};

void Network::propagate_backward(int target_index, double learning_rate) {
    this->compute_output_error(target_index);
    this->compute_hidden_layer_errors();
    this->update_velocities(learning_rate, target_index);
};

void Network::compute_output_error(int target_index) {
    this->ol_error += g_prime(this->ol_neuron.get_net_input()) * (this->training_data.targets[target_index] - this->ol_neuron.get_state());
};

void Network::compute_hidden_layer_errors() {
    for (size_t j = 0; j < this->num_hl_neurons; j++) {
        hl_errors[j] += this->ol_error * this->ol_weights[j] * g_prime(this->hl_neurons[j].get_net_input());
    }
}

void Network::update_velocities(double learning_rate, int target_index) {
    this->old_hl_velocity = this->hl_velocity;
    this->old_ol_velocity = this->ol_velocity;
    this->old_hl_bias_velocity = this->hl_bias_velocity;
    this->old_ol_bias_velocity = this->ol_bias_velocity;

    for (size_t m = 0; m < this->num_hl_neurons; m++) {
        for (size_t n = 0; n < this->num_inputs; n++) {
            this->hl_velocity[m][n] = learning_rate * this->hl_errors[m] * this->training_data.inputs[target_index][n];
        }

        this->ol_velocity[m] = learning_rate * this->ol_error * this->hl_neurons[m].get_state();
        
        this->hl_bias_velocity[m] = learning_rate * this->hl_errors[m];
    }

    this->ol_bias_velocity = learning_rate * this->ol_error;
}

void Network::update_weights_and_biases(double momentum, size_t batch_size) {
    this->ol_error /= static_cast<double>(batch_size);

    for (size_t m = 0; m < this->num_hl_neurons; m++) {
        this->hl_errors[m] /= static_cast<double>(batch_size);

        for (size_t n = 0; n < this->num_inputs; n++) {
            this->hl_weights[m][n] += this->hl_velocity[m][n] + momentum * this->old_hl_velocity[m][n];
        }

        this->ol_weights[m] += this->ol_velocity[m] + momentum * this->old_ol_velocity[m];

        this->hl_biases[m] -= this->hl_bias_velocity[m] + momentum * this->old_hl_bias_velocity[m];
        this->ol_bias -= this->ol_bias_velocity + momentum * this->old_ol_bias_velocity;
    }

    this->ol_error = 0.0;
    this->hl_errors.assign(this->num_hl_neurons, 0.0);
}

void Network::train(double learning_rate, double momentum, size_t batch_size, size_t num_epochs, bool SGD_true) {
    double C_min = 1.0;

    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<size_t> distribution(0, this->num_patterns - 1);

    for (size_t i = 0; i < num_epochs; i++) {
        for (size_t j = 0; j < this-> num_patterns; j++) {

            if (SGD_true) {
                j = distribution(g);
            }

            this->propagate_forward(this->training_data.inputs[j]);
            this->propagate_backward(j, learning_rate);

            if (SGD_true) {
                this->update_weights_and_biases(momentum, batch_size);
            } else if ((j + 1) % batch_size == 0) {
                this->update_weights_and_biases(momentum, batch_size);
            }
        }

        this->validate();

        if (this->C < C_min) {
            C_min = this->C;
        }

        if (this->C < 0.12) {
            std::cout << "C < 0.12!!!!!!!!!!!!!!!!!!" << std::endl;
            break;
        }
    }

    std::cout << "C_min: " << C_min << std::endl;
}

void Network::validate() {
    this->C = 0.0;
    this->H = 0.0;

    for (size_t i = 0; i < this->num_patterns; i++) {
        this->propagate_forward(this->training_data.inputs[i]);
        H += std::pow(this->training_data.targets[i] - this->get_output(), 2);
    }

    H /= 2.0;

    for (size_t i = 0; i < this->num_validation_patterns; i++) {
        this->propagate_forward(this->validation_data.inputs[i]);

        int classification = (this->get_output() > 0) ? 1 : -1;
        C += std::abs(classification - this->validation_data.targets[i]);
    }

    C /= 2 * static_cast<double>(this->num_validation_patterns);

    // std::cout << "C: " << C << " H: " << H << std::endl;
}