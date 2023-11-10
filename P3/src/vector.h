#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

template<typename T>
class Vector {
private:
    std::vector<T> data;
public:
    // Constructor
    explicit Vector(size_t size, const T& value = T())
        : data(size, value) {}

    Vector(std::initializer_list<T> init_list)
        : data(init_list) {
    }

    Vector() : data() {}

    // Access
    T& operator[](size_t index) {
        if (index >= data.size()) {
            throw std::out_of_range("Index out of range.");
        }

        return data[index];
    }

    // Read-only access
    const T& operator[](size_t index) const {
        if (index >= data.size()) {
            throw std::out_of_range("Index out of range.");
        }

        return data[index];
    }

    // Vector addition
    Vector operator+(const Vector& other) const {
        if (data.size() != other.data.size())
            throw std::out_of_range("Vector dimensions do not match for addition.");

        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    Vector& operator+=(const Vector& other) {
        if (data.size() != other.data.size())
            throw std::out_of_range("Vector dimensions do not match for addition.");

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other[i];
        }
        return *this;
    }

    // Vector subtraction
    Vector operator-(const Vector& other) const {
        if (data.size() != other.data.size())
            throw std::out_of_range("Vector dimensions do not match for subtraction.");

        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    Vector& operator-=(const Vector& other) {
        if (data.size() != other.data.size())
            throw std::out_of_range("Vector dimensions do not match for subtraction.");

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= other[i];
        }
        return *this;
    }

    Vector& operator*=(const T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= scalar;
        }
        return *this;
    }

    Vector& in_place_elementwise_multiplication(const Vector& other, Vector& result) {
        if (data.size() != other.data.size())
            throw std::out_of_range("Vector dimensions do not match for element-wise multiplication.");

        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * other[i];
        }
        
        return result;
    }

    // Element-wise multiplication
    Vector operator%(const Vector& other) const {
        if (data.size() != other.data.size())
            throw std::out_of_range("Vector dimensions do not match for element-wise multiplication.");

        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * other[i];
        }
        return result;
    }

    Vector& apply_function(T (*function)(T)) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = function(data[i]);
        }
        return *this;
    }

    // Size of the vector
    size_t size() const {
        return data.size();
    }

    void print() const {
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }

    void reset() {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = T();
        }
    }

    double sum() const {
        double sum = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
        return sum;
    }
};

template<typename T>
class VectorCollection {
    private:
        std::vector<Vector<T>> vectors;
        size_t num_vectors;
    public:
        VectorCollection(size_t num_vectors = 0) {
            vectors = std::vector<Vector<T>>(num_vectors);
            this->num_vectors = num_vectors;
        }

        // Get vector at a specific index
        Vector<T>& operator[](size_t index) {
            return vectors[index];
        }

        const Vector<T>& operator[](size_t index) const {
            return vectors[index];
        }
};