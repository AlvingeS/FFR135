#pragma once

#include "vector.h"
#include <vector>
#include <stdexcept>

template<typename T>
class Matrix {
private:
    size_t rows, cols;
    std::vector<Vector<T>> data;
public:
    // Constructor
    Matrix(size_t _rows, size_t _cols, const T& value = T())
        : rows(_rows), cols(_cols), data(_rows, Vector<T>(_cols, value)) {}

    Matrix() : rows(0), cols(0), data() {}

    // Access
    Vector<T>& operator[](size_t index) {
        if (index >= rows) {
            throw std::out_of_range("Index out of range.");
        }

        return data[index];
    }

    // Read-only access
    const Vector<T>& operator[](size_t index) const {
        if (index >= rows) {
            throw std::out_of_range("Index out of range.");
        }

        return data[index];
    }

    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::out_of_range("Matrix dimensions do not match for addition.");

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] + other[i][j];
            }
        }
        return result;
    }

    Matrix operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols)
            throw std::out_of_range("Matrix dimensions do not match for addition.");

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] += other[i][j];
            }
        }
        return *this;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::out_of_range("Matrix dimensions do not match for addition.");

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] - other[i][j];
            }
        }
        return result;
    }

    Matrix operator -= (const Matrix& other) {
        if (rows != other.rows || cols != other.cols)
            throw std::out_of_range("Matrix dimensions do not match for addition.");

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                data[i][j] -= other[i][j];
            }
        }
        return *this;
    }

    // Matrix scalar multiplication
    Matrix operator*(const T& scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw std::out_of_range("Matrix dimensions do not match for multiplication.");

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result[i][j] += data[i][k] * other[k][j];
                }
            }
        }
        return result;
    }

    // Matrix to vector multiplication
    Vector<T> operator*(const Vector<T>& other) const {
        if (cols != other.size())
            throw std::out_of_range("Matrix and vector dimensions do not match for multiplication.");

        Vector<T> result(rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.size(); ++j) {
                result[i] += data[i][j] * other[j];
            }
        }
        return result;
    }

    // Element-wise multiplication
    Matrix operator%(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::out_of_range("Matrix dimensions do not match for element-wise multiplication.");

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                result[i][j] = data[i][j] * other[i][j];
            }
        }
        return result;
    }

    static Matrix outer_product(const Vector<T>& v1, const Vector<T>& v2) {
        size_t n = v1.size();
        size_t m = v2.size();

        Matrix result(n, m, T());
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                result[i][j] = v1[i] * v2[j];
            }
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix transposed(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed[j][i] = data[i][j];
            }
        }
        return transposed;
    }

    // Apply function element-wise
    Matrix apply_function(T (*function)(T)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = function(data[i][j]);
            }
        }
        return result;
    }

    // Dimensions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            data[i].print();
        }
    }

    double mean() const {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            sum += data[i].sum();
        }
        return sum / (rows * cols);
    }
};

template<typename T>
class MatrixCollection {
private:
    std::vector<Matrix<T>> matrices;
    size_t num_matrices;
public:
    MatrixCollection(size_t num_matrices = 0) {
        matrices = std::vector<Matrix<T>>(num_matrices);
        this->num_matrices = num_matrices;
    }

    // Get matrix at a specific index
    Matrix<T>& operator[](size_t index) {
        if (index >= num_matrices) {
            throw std::out_of_range("Index out of range.");
        }

        return matrices[index];
    }

    const Matrix<T>& operator[](size_t index) const {
        if (index >= num_matrices) {
            throw std::out_of_range("Index out of range.");
        }

        return matrices[index];
    }

    void print_dims() const {
        for (size_t i = 0; i < num_matrices; ++i) {
            std::cout << matrices[i].getRows() << " " << matrices[i].getCols() << std::endl;
        }
    }
};

struct Data {
    Matrix<double> inputs;
    Matrix<double> targets;
};