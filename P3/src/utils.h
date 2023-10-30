#pragma once

#include <vector>
#include <iostream>
#include <string>

typedef std::vector<double> double_vector;
typedef std::vector<int> int_vector;
typedef std::vector<double_vector> double_matrix;
typedef std::vector<double_matrix> double_tensor;

struct Data {
    double_matrix inputs;
    double_matrix targets;
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

template <typename T>
T mean_of_matrix(std::vector<std::vector<T>> &matrix) {
    T sum = 0;
    size_t num_elems = 0;

    for (const auto &row : matrix) {
        for (const auto &elem : row) {
            sum += elem;
            num_elems++;
        }
    }

    return sum / static_cast<T>(num_elems);
}

template <typename T>
T mean_of_vector(std::vector<T> &vec) {
    T sum = 0;
    size_t num_elems = 0;

    for (const auto &elem : vec) {
        sum += elem;
        num_elems++;
    }

    return sum / static_cast<T>(num_elems);
}



