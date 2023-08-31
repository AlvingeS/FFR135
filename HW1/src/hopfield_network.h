#pragma once
#include<vector>
#include<string>
#include "neuron.h"

typedef std::vector<int> state;
typedef std::vector<state> state_vector;
typedef std::vector<std::vector<double>> W;

class HopfieldNetwork {
    public:
        HopfieldNetwork();
        state get_state();
        void train(const state_vector patterns);
        void feed_distorted_pattern(const state distorted_pattern);
        void recall(bool print = false);
        int classify_state(const state_vector patterns);
        
        void print_weights();
        void print_state();
    private:
        const size_t num_neurons = 160;
        const size_t num_columns = 10;
        std::vector<Neuron> neurons;
        W weights;

        int calculate_state_differences(const state current_state, const state new_state);
        void update_neurons(const bool print);
        std::string convert_for_printing(int state);
};