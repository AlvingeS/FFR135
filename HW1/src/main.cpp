#include "hopfield_network.h"
#include<vector>
#include<string>
#include <iostream>
#include <fstream>
#include <sstream>

vector<vector<int>> parsePatterns(const string& filename) {
    vector<vector<int>> patterns;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        vector<int> vector;
        stringstream ss(line);
        string token;

        while (getline(ss, token, ',')) {
            vector.push_back(stoi(token));
        }

        patterns.push_back(vector);
    }

    return patterns;
}

void printPatterns(vector<vector<int>> patterns) {
    for (const auto& pattern : patterns) {
        cout << "Pattern: ";
        for (const auto& value : pattern) {
            cout << value << " ";
        }
        cout << endl;
    }
}

int main() {
    vector<vector<int>> patterns = parsePatterns("patterns.txt");
    vector<vector<int>> distorted_patterns = parsePatterns("distorted_patterns.txt");

    HopfieldNetwork hopfield_network = HopfieldNetwork(10*16);
    hopfield_network.train(patterns);
    vector<int> classified_numbers = hopfield_network.recall(distorted_patterns);
    // hopfield_network.print_weights();
    
    hopfield_network.feed_distorted_pattern(distorted_patterns[2]);
    hopfield_network.print_state(10);
    hopfield_network.update_neurons();
    cout << "----------------------------" << endl;
    hopfield_network.print_state(10);
    return 0;
}