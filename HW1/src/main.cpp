#include "hopfield_network.h"
#include<vector>
#include<string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

// Converts patterns from a file to a vectors of integers
state_vector parse_patterns(const std::string& filename) {
    state_vector patterns;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        state new_state;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            new_state.push_back(std::stoi(token));
        }

        patterns.push_back(new_state);
    }

    return patterns;
}

// Converts a state to a pattern string
std::string convert_state_to_scheme(const state& state) {
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

int main(int argc, char *argv[]) {

    // Convert the argument to an integer
    int distorted_pattern_index = std::atoi(argv[1]);

    // Load patterns from file
    state_vector patterns = parse_patterns("patterns.txt");
    state_vector distorted_patterns = parse_patterns("distorted_patterns.txt");

    // Create and train hopfield network with 10*16 neurons
    HopfieldNetwork hopfield_network = HopfieldNetwork();
    hopfield_network.train(patterns);
    
    // Feed distorted pattern to hopfield network, setting the initial state
    hopfield_network.feed_distorted_pattern(distorted_patterns[distorted_pattern_index]);

    // Update state of network until it converges
    hopfield_network.recall();

    // Print the steady state
    std::cout << convert_state_to_scheme(hopfield_network.get_state()) << std::endl;
    
    // Print the classification of the steady state
    std::cout << hopfield_network.classify_state(patterns) << std::endl;

    return 0;
}