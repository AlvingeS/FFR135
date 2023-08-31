#include "hopfield_network.h"
#include<vector>
#include<string>
#include <iostream>
#include <fstream>
#include <sstream>

// Converts patterns from a file to a vectors of integers
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

// Converts a state to a pattern string
std::string convert_state_to_scheme(const std::vector<int>& state) {
    std::stringstream ss;
    ss << '[';

    for (size_t i = 0; i < state.size(); ++i) {
        if (i % 10 == 0) {
            ss << '[';
        }

        ss << state[i];

        if ((i + 1) % 10 == 0 && i != 0) {
            ss << ']';
            if (i != state.size() - 1) {
                ss << ", ";
            }
        } else if (i < state.size() - 1) {
            ss << ", ";
        }
    }

    if (state.size() % 10 != 0) {
        ss << ']';
    }
    
    ss << ']';  // The last ']' is to close the overall bracket
    return ss.str();
}

int main() {
    vector2d_int patterns = parse_patterns("patterns.txt");
    vector2d_int distorted_patterns = parse_patterns("distorted_patterns.txt");

    HopfieldNetwork hopfield_network = HopfieldNetwork(10*16);
    hopfield_network.train(patterns);
    std::vector<int> classified_numbers = hopfield_network.recall(distorted_patterns);
    // hopfield_network.print_weights();
    
    hopfield_network.feed_distorted_pattern(distorted_patterns[2]);
    hopfield_network.update_neurons();
    std::cout << convert_state_to_scheme(hopfield_network.get_state()) << std::endl;

    return 0;
}