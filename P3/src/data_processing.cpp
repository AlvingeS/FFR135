#include "matrix.h"
#include <fstream>
#include <sstream>
#include <utility>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>


Data read_csv(const std::string &filename, size_t num_inputs, size_t num_outputs) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
    }

    // Calculate how many rows there are in the file
    size_t num_rows = 0;
    std::string line;
    while (std::getline(file, line)) {
        num_rows++;
    }
    
    Data data = {Matrix(num_rows, num_inputs, 0.0), Matrix(num_rows, num_outputs, 0.0)};

    // Reset the file pointer to the beginning of the file
    file.clear();
    file.seekg(0);

    // Read the first line to determine the number of fields
    std::string first_line;
    std::getline(file, first_line);
    size_t num_commas = std::count(first_line.begin(), first_line.end(), ',');
    size_t total_fields = num_commas + 1;

    if (total_fields != num_inputs + num_outputs) {
        std::cerr << "Error: Number of fields in CSV (" << total_fields 
                  << ") does not match num_inputs (" << num_inputs 
                  << ") + num_outputs (" << num_outputs << ")" << std::endl;
        return data;
    }

    size_t row_ind = 0;

    while (std::getline(file, line)) {
        std::istringstream s(line);
        std::string field;

        for (size_t i = 0; i < num_inputs; ++i) {
            std::getline(s, field, ',');
            data.inputs[row_ind][i] = std::stod(field);
        }

        for (size_t i = 0; i < num_outputs; ++i) {
            std::getline(s, field, ',');
            data.targets[row_ind][i] = std::stod(field);
        }

        row_ind++;
    }

    file.close();
    return data;
}

void normalize_input_data(Data &training_data, Data &data_to_be_normalized) {
    size_t num_samples = training_data.inputs.getRows();
    size_t num_features = training_data.inputs.getCols();
    Vector<double> mean(num_features, 0.0);
    Vector<double> std_dev(num_features, 0.0);
    
    // Calculate the mean of each feature
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < num_features; ++j) {
            mean[j] += training_data.inputs[i][j];
        }
    }

    for (size_t j = 0; j < num_features; ++j) {
        mean[j] /= num_samples;
    }
    
    // Calculate the standard deviation of each feature
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < num_features; ++j) {
            std_dev[j] += std::pow(training_data.inputs[i][j] - mean[j], 2);
        }
    }

    for (size_t j = 0; j < num_features; ++j) {
        std_dev[j] = std::sqrt(std_dev[j] / num_samples);
    }
    
    // Normalize the data
    for (size_t i = 0; i < data_to_be_normalized.inputs.getRows(); i++) {
        for (size_t j = 0; j < num_features; ++j) {
           data_to_be_normalized.inputs[i][j] = (data_to_be_normalized.inputs[i][j] - mean[j]) / std_dev[j];
        }
    }
}

void shuffle_data(Data& data) {
    size_t num_samples = data.inputs.getRows();
    
    std::random_device rd;
    std::default_random_engine engine(rd());
    
    std::vector<size_t> indices(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }
    
    std::shuffle(indices.begin(), indices.end(), engine);
    
    Matrix<double> new_inputs(data.inputs.getRows(), data.inputs.getCols(), 0.0);
    Matrix<double> new_targets(data.targets.getRows(), data.targets.getCols(), 0.0);
    
    for (size_t i = 0; i < num_samples; ++i) {
        new_inputs[i] = data.inputs[indices[i]];
        new_targets[i] = data.targets[indices[i]];
    }

    data.inputs = new_inputs;
    data.targets = new_targets;
}