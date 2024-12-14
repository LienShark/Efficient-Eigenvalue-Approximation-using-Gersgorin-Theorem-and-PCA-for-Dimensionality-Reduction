#pragma once

#include <vector>
#include <iostream>
#include <thread>
#include <tuple>
#include <algorithm>
#include <cmath>
using namespace std;
enum class MultiplicationMethod {
    Naive,
    Strassen,
    TiledNaive
};

class Matrix{
private:
    size_t m_nrow, m_ncol;
    double *m_buffer;
public:
    Matrix();
    Matrix(size_t nrow, size_t ncol);
    Matrix(size_t nrow, size_t ncol, double val);
    Matrix(size_t nrow, size_t ncol,const vector<double> &v);
    Matrix(const Matrix &m);
    ~Matrix();

    size_t index(size_t i, size_t j) const;
    size_t nrow() const;
    size_t ncol() const;
    double* get_buffer() const;

    double   operator() (size_t row, size_t col) const;
    double & operator() (size_t row, size_t col);
    bool operator==(const Matrix &m);  
    Matrix operator+(const Matrix &m);
    Matrix operator-(const Matrix &m);
    Matrix& operator=(const Matrix &m);
    Matrix transpose() const;
    // std::tuple<Matrix, Matrix, Matrix> jacobi_eigen(const Matrix &m);
    // std::tuple<Matrix, Matrix, Matrix> svd_jacobi(const Matrix &m);
    // Matrix multiply(const Matrix &m) const;
    Matrix sqrt() const;
    Matrix inverse() const;
    Matrix operator*(double scalar) const;
};
Matrix matrix_multiply_naive(Matrix const &m1, Matrix const &m2);
Matrix submatrix(const Matrix &matrix, size_t row_start, size_t col_start, size_t row_end, size_t col_end);
// 在 SVD.hpp 或 Matrix.hpp 中
Matrix strassen_recursive(const Matrix &A, const Matrix &B, size_t block_size);
Matrix strassen_matrix_multiply(const Matrix &m1, const Matrix &m2, size_t block_size);
