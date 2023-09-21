#include "utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>

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