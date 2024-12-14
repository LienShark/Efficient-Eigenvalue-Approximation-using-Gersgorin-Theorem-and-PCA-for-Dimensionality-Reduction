#ifndef SVD_HPP
#define SVD_HPP

#include "Matrix.hpp"

std::tuple<Matrix, Matrix, Matrix> jacobi_eigen(const Matrix &m);
std::tuple<Matrix, Matrix, Matrix> svd_jacobi(const Matrix &A,bool use_strassen = false);
Matrix compute_null_space(const Matrix &A);
Matrix concatenate_columns(const Matrix &A, const Matrix &B);

#endif
