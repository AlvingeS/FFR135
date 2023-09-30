#include "utils.h"
#include <fstream>
#include <sstream>
#include <utility>
#include <cmath>
#include <random>
#include <algorithm>

Data read_csv(const std::string &filename) {
    std::ifstream file(filename);

    Data data = {double_matrix(), double_vector()};

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        double_vector row;
        std::istringstream s(line);
        std::string field;

        // Read first input
        std::getline(s, field, ',');
        row.push_back(std::stod(field));

        // Read second input
        std::getline(s, field, ',');
        row.push_back(std::stod(field));

        data.inputs.push_back(row);

        // Read target
        std::getline(s, field, ',');
        data.targets.push_back(std::stoi(field));
    }

    file.close();
    return data;
}

void normalize_input_data(Data &training_data, Data &data_to_be_normalized) {
    double_vector mean(2, 0.0);
    double_vector std_dev(2, 0.0);
    size_t num_samples = training_data.inputs.size();
    
    // Calculate the mean of each feature
    for (const auto &row : training_data.inputs) {
        for (size_t i = 0; i < 2; ++i) {
            mean[i] += row[i];
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        mean[i] /= num_samples;
    }
    
    // Calculate the standard deviation of each feature
    for (const auto &row : training_data.inputs) {
        for (size_t i = 0; i < 2; ++i) {
            std_dev[i] += std::pow(row[i] - mean[i], 2);
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        std_dev[i] = std::sqrt(std_dev[i] / num_samples);
    }
    
    // Normalize the data
    for (auto &row : data_to_be_normalized.inputs) {
        for (size_t i = 0; i < 2; ++i) {
            row[i] = (row[i] - mean[i]) / std_dev[i];
        }
    }
}

void shuffle_data(Data& data) {
    size_t data_size = data.inputs.size();
    
    std::random_device rd;
    std::default_random_engine engine(rd());
    
    std::vector<size_t> indices(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        indices[i] = i;
    }
    
    std::shuffle(indices.begin(), indices.end(), engine);
    
    double_matrix new_inputs(data_size);
    double_vector new_targets(data_size);
    
    for (size_t i = 0; i < data_size; ++i) {
        new_inputs[i] = data.inputs[indices[i]];
        new_targets[i] = data.targets[indices[i]];
    }
    
    data.inputs.swap(new_inputs);
    data.targets.swap(new_targets);
}