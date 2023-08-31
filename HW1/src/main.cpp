#include "hopfield_network.h"
#include<vector>
#include<string>
#include <iostream>
#include <fstream>
#include <sstream>

vector2d_int parse_patterns(const std::string& filename) {
    vector2d_int patterns;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<int> vector;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            vector.push_back(std::stoi(token));
        }

        patterns.push_back(vector);
    }

    return patterns;
}

void print_patterns(vector2d_int patterns) {
    for (const auto& pattern : patterns) {
        std::cout << "Pattern: ";
        for (const auto& value : pattern) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    vector2d_int patterns = parse_patterns("patterns.txt");
    vector2d_int distorted_patterns = parse_patterns("distorted_patterns.txt");

    HopfieldNetwork hopfield_network = HopfieldNetwork(10*16);
    hopfield_network.train(patterns);
    std::vector<int> classified_numbers = hopfield_network.recall(distorted_patterns);
    // hopfield_network.print_weights();
    
    hopfield_network.feed_distorted_pattern(distorted_patterns[2]);
    hopfield_network.print_state(10);
    hopfield_network.update_neurons();
    std::cout << "----------------------------" << std::endl;
    hopfield_network.print_state(10);
    return 0;
}