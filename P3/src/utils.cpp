#include "utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <cmath>

std::pair<std::vector<std::vector<double>>, std::vector<int>> read_csv(const std::string &filename) {
    std::ifstream file(filename);

    std::vector<std::vector<double>> temp_inputs;
    std::vector<int> temp_targets;

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return std::make_pair(temp_inputs, temp_targets);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream s(line);
        std::string field;

        // Read first input
        std::getline(s, field, ',');
        row.push_back(std::stod(field));

        // Read second input
        std::getline(s, field, ',');
        row.push_back(std::stod(field));

        temp_inputs.push_back(row);

        // Read target
        std::getline(s, field, ',');
        temp_targets.push_back(std::stoi(field));
    }

    file.close();
    return std::make_pair(temp_inputs, temp_targets);
}

std::vector<std::vector<double>> normalize_data(const std::vector<std::vector<double>> &input_data) {
    std::vector<double> mean(2, 0.0);
    std::vector<double> std_dev(2, 0.0);
    size_t num_samples = input_data.size();
    
    // Calculate the mean of each feature
    for (const auto &row : input_data) {
        for (size_t i = 0; i < 2; ++i) {
            mean[i] += row[i];
        }
    }
    for (size_t i = 0; i < 2; ++i) {
        mean[i] /= num_samples;
    }
    
    // Calculate the standard deviation of each feature
    for (const auto &row : input_data) {
        for (size_t i = 0; i < 2; ++i) {
            std_dev[i] += std::pow(row[i] - mean[i], 2);
        }
    }
    for (size_t i = 0; i < 2; ++i) {
        std_dev[i] = std::sqrt(std_dev[i] / num_samples);
    }
    
    // Create a new matrix to store the normalized data
    std::vector<std::vector<double>> normalized_data = input_data;
    
    // Normalize the data
    for (auto &row : normalized_data) {
        for (size_t i = 0; i < 2; ++i) {
            row[i] = (row[i] - mean[i]) / std_dev[i];
        }
    }
    
    return normalized_data;
}