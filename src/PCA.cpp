#include "include/PCA.hpp"
//計算每一行的平均
Matrix compute_mean(const Matrix &data) {
    size_t rows = data.nrow();
    size_t cols = data.ncol();
    Matrix mean(1, cols, 0.0);

    for (size_t j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            sum += data(i, j);
        }
        mean(0, j) = sum / rows;
    }
    return mean;
}

//計算每一行減完平均的樣子(準備中心化)
Matrix center_data(const Matrix &data, const Matrix &mean) {
    size_t rows = data.nrow();
    size_t cols = data.ncol();
    Matrix centered_data = data;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            centered_data(i, j) -= mean(0, j);
        }
    }
    return centered_data;
}

// Perform PCA
std::tuple<Matrix, Matrix> PCA(const Matrix &data, size_t num_components,bool use_strassen) {
    Matrix mean = compute_mean(data);
    Matrix centered_data = center_data(data, mean);
    //計算convariance matrix
    Matrix covariance;
    if(use_strassen){
        covariance = strassen_matrix_multiply(centered_data.transpose(), centered_data , 64);
    }else{
        covariance = matrix_multiply_naive(centered_data.transpose(), centered_data);
    }
    covariance = covariance * (1.0 / data.nrow());

    Matrix U, S, V;
    std::tie(U, S, V) = svd_jacobi(covariance);

    Matrix principal_components(U.nrow(), num_components);
    for (size_t i = 0; i < U.nrow(); ++i) {
        for (size_t j = 0; j < num_components; ++j) {
            principal_components(i, j) = U(i, j);
        }
    }

    // double total_variance = 0.0;
    // for (size_t i = 0; i < S.nrow(); ++i) {
    //     total_variance += S(i, i);
    // }

    // Matrix explained_variance_ratio(1, num_components);
    // for (size_t i = 0; i < num_components; ++i) {
    //     explained_variance_ratio(0, i) = S(i, i) / total_variance;
    // }
    Matrix explained_variance(1, num_components);
    for (size_t i = 0; i < num_components; ++i) {
        explained_variance(0, i) = S(i, i); // 使用原始特徵值
    }
    return {principal_components, explained_variance};
}
