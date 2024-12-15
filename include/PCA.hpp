#ifndef PCA_HPP
#define PCA_HPP

#include "Matrix.hpp"
#include "SVD.hpp"
#include <tuple>

Matrix compute_mean(const Matrix &data);
Matrix center_data(const Matrix &data, const Matrix &mean);
std::tuple<Matrix, Matrix> PCA(const Matrix &data, size_t num_components,bool use_strassen = false);

#endif
