#ifndef SIMD_OPTIMIZED_HPP
#define SIMD_OPTIMIZED_HPP

#include "Matrix.hpp"

// SIMD 優化的矩陣乘法函數聲明
Matrix simd_matrix_multiply(const Matrix &m1, const Matrix &m2);
std::tuple<Matrix, Matrix, Matrix> jacobi_eigen_simd(const Matrix &A);
std::tuple<Matrix, Matrix, Matrix> svd_jacobi_simd(const Matrix &A);
#endif
