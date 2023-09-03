#include "hopfield_network.h"
#include "utils.h"
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

int main(int argc, char *argv[]) {
    // Convert the commandline argument to an integer
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