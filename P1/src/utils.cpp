#include "utils.h"
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>


// Prints the weights of the hopfield network (For debugging purposes)
void print_weights(std::vector<std::vector<double>> weights) {
    for (const auto& row : weights) {
        for (const auto& weight : row) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
    }
}

// Converts a state to an easy-to-read string
std::string convert_for_printing(int state) {
    if (state == 1) {
        return "X";
    } else {
        return " ";
    }
}

// Prints the state of the neurons in the hopfield network
// Flushes the screen to see the evolution of the state
void print_state(size_t num_neurons, size_t num_columns, std::vector<int> states) {
    std::cout << "\033[2J\033[1;1H";  // Clear screen and move cursor to top-left corner

    int counter = 0;

    for (size_t i = 0; i < num_neurons; i++) {
        std::cout << convert_for_printing(states[i]) << " ";

        if (++counter % num_columns == 0) {
            std::cout << std::endl;
            counter = 0;
        }
    }
    std::cout << std::endl;
    
    usleep(5000);
}

// Converts patterns from a file to a vectors of integers
std::vector<std::vector<int>> parse_patterns(const std::string& filename) {
    std::vector<std::vector<int>> patterns;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<int> new_state;
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