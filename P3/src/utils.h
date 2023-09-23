#pragma once

#include <vector>
#include <iostream>
#include <string>

typedef std::vector<double> double_vector;
typedef std::vector<int> int_vector;
typedef std::vector<double_vector> double_matrix;
struct Data {
    double_matrix inputs;
    double_vector targets;
};

template <typename T>
void print_matrix(std::vector<std::vector<T>> &matrix) {
    for (const auto &row : matrix) {
        for (const auto &elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void print_vector(std::vector<T> &vec) {
    for (const auto &elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}


