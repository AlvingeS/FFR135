#pragma once

#include "neuron.h"
#include "utils.h"
#include <cstddef>
#include <vector>
#include <cstdint>
#include <cmath>

typedef std::vector<Neuron> neuron_vector;
typedef std::vector<neuron_vector> neuron_matrix;

struct arch_struct {
    size_t num_inputs;
    int_vector hl_sizes;
    size_t num_outputs;

    int num_hls() const {
        return this->hl_sizes.size();
    }
};

class Network {
    public:
        Network(arch_struct arch, Data training_data, Data validation_data);
        
        double_vector get_output() {
            double_vector output(this->layer_heights[L], 0.0);

            for (size_t i = 0; i < this->layer_heights[L]; i++) {
                output[i] = this->neurons[L - 1][i].get_state();
            }

            return output;
        }
        
        void train(double learning_rate, double momentum, size_t batch_size, size_t num_epoch, bool measure_H, bool verbose);

        void export_validation_results();

    private:
        
        double g_prime(double x) {
            return 1 - std::pow(std::tanh(x), 2);
        }
        
        void propagate_forward(const std::vector<double> &input_signals);
        void compute_errors(int target_index);
        void update_velocities(double learning_rate, int target_index, size_t batch_size);
        void update_weights_and_biases(double momentum);
        void validate(size_t epoch, bool measure_H, bool verbose);

        size_t num_patterns;
        size_t num_validation_patterns;
        
        arch_struct arch;
        int_vector layer_heights;
        size_t L;
        size_t offset = 1;
        
        // Data storage
        Data training_data;
        Data validation_data;

        double_tensor weights;
        double_matrix biases;

        double_tensor velocities_w;
        double_matrix velocities_b;
        double_tensor velocities_w_old;
        double_matrix velocities_b_old;
        
        neuron_matrix neurons;

        double_matrix cumulative_errors;
        double_tensor cumulative_products;

        double C = 0.0;
        double H = 0.0;
};