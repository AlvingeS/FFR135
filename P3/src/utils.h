#pragma once

#include <vector>
#include <iostream>

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