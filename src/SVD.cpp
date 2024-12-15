#include "include/SVD.hpp"
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <algorithm>

std::tuple<Matrix, Matrix, Matrix> jacobi_eigen(const Matrix &m) {
    if (m.nrow() != m.ncol()) {
        throw std::invalid_argument("Jacobi SVD requires a square matrix.");
    }

    const size_t n = m.nrow();
    Matrix U(n, n, 0.0);
    Matrix V(n, n, 0.0);
    Matrix S = m; // 複製輸入矩陣

    const double epsilon = 1e-10; // 收斂閾值
    const size_t max_iter = 1000000; //增加迭帶次數，否則大型矩陣會出錯

    // 初始化 U 和 V 為單位矩陣
    for (size_t i = 0; i < n; ++i) {
        U(i, i) = 1.0;
        V(i, i) = 1.0;
    }

    for (size_t iter = 0; iter < max_iter; ++iter) {
        double max_offdiag = 0.0;
        size_t p = 0, q = 0;

        // 找到絕對值最大的非對角元素
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (std::abs(S(i, j)) > max_offdiag) {
                    max_offdiag = std::abs(S(i, j));
                    p = i;
                    q = j;
                }
            }
        }

        if (max_offdiag < epsilon) {
            break; // 已收斂
        }

        // 計算旋轉角度
        double tau = (S(q, q) - S(p, p)) / (2.0 * S(p, q));
        double t = (tau > 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        double cos_phi = 1.0 / std::sqrt(1.0 + t * t);
        double sin_phi = t * cos_phi;

        // 更新 S
        for (size_t i = 0; i < n; ++i) {
            double sp = cos_phi * S(i, p) - sin_phi * S(i, q);
            double sq = sin_phi * S(i, p) + cos_phi * S(i, q);
            S(i, p) = sp;
            S(i, q) = sq;
        }
        for (size_t j = 0; j < n; ++j) {
            double sp = cos_phi * S(p, j) - sin_phi * S(q, j);
            double sq = sin_phi * S(p, j) + cos_phi * S(q, j);
            S(p, j) = sp;
            S(q, j) = sq;
        }

        // 更新 U
        for (size_t i = 0; i < n; ++i) {
            double up = cos_phi * U(i, p) - sin_phi * U(i, q);
            double uq = sin_phi * U(i, p) + cos_phi * U(i, q);
            U(i, p) = up;
            U(i, q) = uq;
        }

        // 更新 V
        for (size_t i = 0; i < n; ++i) {
            double vp = cos_phi * V(i, p) - sin_phi * V(i, q);
            double vq = sin_phi * V(i, p) + cos_phi * V(i, q);
            V(i, p) = vp;
            V(i, q) = vq;
        }
    }

    // 排序奇異值，並對應調整 U 和 V
    std::vector<std::pair<double, size_t>> singular_values;
    for (size_t i = 0; i < n; ++i) {
        singular_values.emplace_back(S(i, i), i);
    }
    std::sort(singular_values.rbegin(), singular_values.rend()); // 降序排序

    Matrix S_sorted(n, n, 0.0);
    Matrix U_sorted(n, n, 0.0);
    Matrix V_sorted(n, n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        size_t idx = singular_values[i].second;
        S_sorted(i, i) = singular_values[i].first;

        for (size_t j = 0; j < n; ++j) {
            U_sorted(j, i) = U(j, idx);
            V_sorted(j, i) = V(j, idx);
        }
    }

    return std::make_tuple(U_sorted, S_sorted, V_sorted);
}


Matrix compute_null_space(const Matrix &A) {
    // Compute the null space of matrix A
    size_t n = A.ncol();
    size_t m = A.nrow();

    Matrix ATA = matrix_multiply_naive(A.transpose(), A);
    Matrix V, S;
    std::tie(V, S, std::ignore) = jacobi_eigen(ATA);

    Matrix null_space(n, n - m, 0.0); // rank+nulity = n
    size_t col_index = 0;

    for (size_t i = m; i < n; ++i) { 
        for (size_t j = 0; j < n; ++j) {
            null_space(j, col_index) = V(j, i);
        }
        col_index++;
    }

    return null_space;
}

Matrix concatenate_columns(const Matrix &A, const Matrix &B) {
    // 檢查行數是否一致
    if (A.nrow() != B.nrow()) {
        throw std::invalid_argument("Row dimensions must match for column concatenation.");
    }

    // 建立新的矩陣，行數與 A 和 B 相同，列數為 A 和 B 的列數之和
    size_t new_ncol = A.ncol() + B.ncol();
    Matrix result(A.nrow(), new_ncol);

    // 複製 A 的數據
    for (size_t i = 0; i < A.nrow(); ++i) {
        for (size_t j = 0; j < A.ncol(); ++j) {
            result(i, j) = A(i, j);
        }
    }

    // 複製 B 的數據
    for (size_t i = 0; i < B.nrow(); ++i) {
        for (size_t j = 0; j < B.ncol(); ++j) {
            result(i, j + A.ncol()) = B(i, j); // 將 B 的列偏移後存入
        }
    }

    return result;
}


std::tuple<Matrix, Matrix, Matrix> svd_jacobi(const Matrix &A,bool use_strassen) {
    size_t m = A.nrow();
    size_t n = A.ncol();

    Matrix U(m,m), S(m,n), V(n,n);

    if (m >= n) {
        // 先求ATA
        Matrix ATA(n,n);
        if(use_strassen){
            ATA = strassen_matrix_multiply(A.transpose(), A, 32);
        }else{
            ATA = matrix_multiply_naive(A.transpose(), A); // ATA = A^T * A
        }
        
        std::tie(V, S, std::ignore) = jacobi_eigen(ATA);

        Matrix S_expanded(m, n, 0.0);//S要是m*n但jacobi只會返回n*n的matrix，要自行添加row數
        for (size_t i = 0; i < n; ++i) {
            S_expanded(i, i) = std::sqrt(S(i, i)); 
        }
        S = S_expanded;

        // U = AVS^-1
        Matrix Sigma_inv(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            if (S_expanded(i, i) > 1e-10) { //避免除0
                Sigma_inv(i, i) = 1.0 / S_expanded(i, i);
            }
        }
        if(use_strassen){
            U = strassen_matrix_multiply(A, strassen_matrix_multiply(V,Sigma_inv,32),32);
        }else{
            U = matrix_multiply_naive(A, matrix_multiply_naive(V, Sigma_inv));
        }
        
        //U不足的從N(AT)的orthogonal basis尋找
        if (m > n) {
            Matrix AT = A.transpose();
            Matrix null_basis = compute_null_space(AT); 
            U = concatenate_columns(U, null_basis);     
        }

    } else {
        Matrix AAT(m,m);
        if(use_strassen){
            AAT = strassen_matrix_multiply(A,A.transpose(),64);
        }else{
            AAT = matrix_multiply_naive(A, A.transpose());
        }
        
        std::tie(U, S, std::ignore) = jacobi_eigen(AAT);

        Matrix S_expanded(m, n, 0.0);
        for (size_t i = 0; i < m; ++i) {
            S_expanded(i, i) = std::sqrt(S(i, i));
        }
        S = S_expanded;

        Matrix Sigma_inv(m, m, 0.0);
        for (size_t i = 0; i < m; ++i) {
            if (S_expanded(i, i) > 1e-10) {
                Sigma_inv(i, i) = 1.0 / S_expanded(i, i);
            }
        }
        if(use_strassen){
            V = strassen_matrix_multiply(A.transpose() , strassen_matrix_multiply(U, Sigma_inv , 64),64);
        }else{
            V = matrix_multiply_naive(A.transpose(), matrix_multiply_naive(U, Sigma_inv));
        }
        
        if (n > m) {
            Matrix A_null = compute_null_space(A); 
            V = concatenate_columns(V, A_null);  
        }
    }
    return {U, S, V};
}


