#pragma once
#include<vector>
#include<string>
#include "neuron.h"

typedef std::vector<std::vector<int>> vector2d_int;
typedef std::vector<std::vector<double>> vector2d_double;

class HopfieldNetwork {
    public:
        HopfieldNetwork(int num_neurons);
        void train(vector2d_int patterns);
        std::vector<int> recall(vector2d_int distorted_patterns);
        const void print_weights();
        const void print_state(int nr_columns);
        void feed_distorted_pattern(std::vector<int> distorted_pattern);
        void update_neurons(bool print);
        std::vector<int> get_state();
        int classify_state(vector2d_int patterns);
    private:
        size_t num_neurons;
        vector2d_double weights;
        std::vector<Neuron> neurons;
        std::string convert_for_printing(int state);
        void check_convergence();
};