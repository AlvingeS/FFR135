#pragma once

#include <unordered_map>
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
        
        void train(double learning_rate, double momentum, size_t batch_size, size_t num_epoch, bool measure_H, bool verbose, double lookup_tol);

    private:
        double lookup(double x, double lookup_tol, double (*function)(double)) {
            int bucket_key = static_cast<int>(x / lookup_tol);

            if (lookup_table.find(bucket_key) != lookup_table.end()) {
                for (const auto &pair : lookup_table[bucket_key]) {
                    if (std::abs(pair.first - x) < lookup_tol) {
                        return pair.second;
                    }
                }
            }

            double result = function(x);
            lookup_table[bucket_key].push_back(std::make_pair(x, result));
            return result;
        }

        static double g(double x) {
            return std::tanh(x);
        }

        static double g_prime(double x) {
            return 1 - std::pow(std::tanh(x), 2);
        }
        
        void propagate_forward(const Vector<double> &input_signals, double lookup_tol);
        void compute_errors(int target_index, double lookup_tol);
        void update_velocities(double learning_rate, int target_index, size_t batch_size);
        void update_weights_and_biases(double momentum);
        void validate(size_t epoch, bool measure_H, bool verbose, double lookup_tol);

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
        MatrixCollection<double> weights_transposed;
        MatrixCollection<double> cumulative_products;
        MatrixCollection<double> velocities_w;
        MatrixCollection<double> velocities_w_old;

        VectorCollection<double> deltas;
        VectorCollection<double> neuron_states;
        VectorCollection<double> net_inputs;
        VectorCollection<double> biases;
        VectorCollection<double> cumulative_errors;
        VectorCollection<double> velocities_b;
        VectorCollection<double> velocities_b_old;

        double scaled_learning_rate;

        Vector<double> output_diff;
        Vector<double> output_element_wise_diff;
        VectorCollection<double> internal_element_wise_diff;

        using Bucket = std::vector<std::pair<double, double>>;
        std::unordered_map<int, Bucket> lookup_table;

        double C = 0.0;
        double H = 0.0;
};