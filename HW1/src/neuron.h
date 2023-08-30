#include<vector>

#pragma once

using namespace std;

class Neuron {
    public:
        Neuron(const vector<double> &initial_weights, const double initial_threshold);
        double get_output(const vector<double> &input);

    private:
        vector<double> weights;
        double threshold;
};