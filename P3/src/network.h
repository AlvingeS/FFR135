#pragma once

#include "utils.h"
#include "matrix.h"
#include <cstddef>
#include <vector>
#include <cstdint>
#include <cmath>

struct arch_struct {
    size_t num_inputs;
    Vector<int> hl_sizes;
    size_t num_outputs;

    int num_hls() const {
        return this->hl_sizes.size();
    }
};

class Network {
    public:
        Network(arch_struct arch, Data training_data, Data validation_data);
        
        Vector<double> get_output() {
            return this->neuron_states[L - offset];
        }
        
        void train(double learning_rate, double momentum, size_t batch_size, size_t num_epoch, bool measure_H, bool verbose);

        void export_validation_results();

    private:
        static double g(double x) {
            return std::tanh(x);
        }

        static double g_prime(double x) {
            return 1 - std::pow(std::tanh(x), 2);
        }
        
        void propagate_forward(const Vector<double> &input_signals);
        void compute_errors(int target_index);
        void update_velocities(double learning_rate, int target_index, size_t batch_size);
        void update_weights_and_biases(double momentum);
        void validate(size_t epoch, bool measure_H, bool verbose);

        size_t num_patterns;
        size_t num_validation_patterns;
        
        size_t offset = 1;
        arch_struct arch;
        Vector<int> layer_heights;
        size_t L;
        
        // Data storage
        Data training_data;
        Data validation_data;

        MatrixCollection<double> weights;
        MatrixCollection<double> cumulative_products;
        MatrixCollection<double> velocities_w;
        MatrixCollection<double> velocities_w_old;

        VectorCollection<double> neuron_states;
        VectorCollection<double> net_inputs;
        VectorCollection<double> biases;
        VectorCollection<double> cumulative_errors;
        VectorCollection<double> velocities_b;
        VectorCollection<double> velocities_b_old;

        double C = 0.0;
        double H = 0.0;
};